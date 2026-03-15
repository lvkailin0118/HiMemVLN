"""
VLM-based Trainer for Vision-Language Navigation
This module provides a VLM-based training/evaluation pipeline that uses
Qwen2-VL for direct visual understanding instead of LLM + RAM text descriptions.

Key differences from base_il_trainer_llm.py:
1. Uses Open_Nav_VLM instead of Open_Nav
2. Passes images directly to VLM for navigation decisions
3. Maintains spatial descriptions as supplementary context
4. Logs VLM-specific metrics and visualizations
"""

import json
import sys
import jsonlines
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI
import logging

# for VLM navigator
from vlnce_baselines.common.navigator.spatialNavigator_vlm import Open_Nav_VLM
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
import numpy as np
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image, images_to_video
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class BaseVLNCETrainerVLM(BaseILTrainer):
    """
    VLM-based trainer for VLN-CE navigation.
    
    This trainer uses Qwen2-VL (or similar VLM) for direct visual understanding,
    replacing the traditional LLM + RAM text description approach.
    """
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        
        # Initialize waypoint predictor
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                '/home/manager/lkl/VLN/CVPR2026/Open-Nav-main/waypoint_prediction/checkpoints/check_val_best_avg_wayscore/check_val_best_avg_wayscore',
                map_location=torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        """
        Generate input data from observations.
        
        Returns:
            tuple: (instruction, image_dict)
                - instruction: navigation instruction text
                - image_dict: dict mapping direction_id -> {'rgb': PIL.Image, 'depth': PIL.Image}
        """
        instruction = observations['instruction']['text']
        image_dict = {} 
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        
        for key in observations.keys():
            image_path = "./image_show/"
            if 'rgb' in key:
                image_path += f"{key}.jpg"
                image = Image.fromarray(observations[key], mode="RGB")
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path, format="JPEG")
                rgb_image_dict[str(rgb_index)] = Image.open(image_path)
                rgb_index += 1
            if 'depth' in key:
                image_path += f"{key}.jpg"
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                else:
                    depth_map = observations[key]
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)).astype(np.uint8)
                image = Image.fromarray(depth_img)
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path)
                depth_image_dict[str(depth_index)] = Image.open(image_path)
                depth_index += 1
                
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict.get(index, rgb_image_dict[index])  # Fallback to RGB if no depth
            }
            
        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        """
        Construct waypoint image dictionaries based on angles.
        
        Maps predicted waypoints to their corresponding direction images.
        """
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        angles = batch_angles[-1]
        
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            
            # Map angle to direction bucket (30 degree increments)
            if 0 < angle_deg <= 30:
                dir_id = '1'
            elif 30 < angle_deg <= 60:
                dir_id = '2'
            elif 60 < angle_deg <= 90:
                dir_id = '3'
            elif 90 < angle_deg <= 120:
                dir_id = '4'
            elif 120 < angle_deg <= 150:
                dir_id = '5'
            elif 150 < angle_deg <= 180:
                dir_id = '6'
            elif 180 < angle_deg <= 210:
                dir_id = '7'
            elif 210 < angle_deg <= 240:
                dir_id = '8'
            elif 240 < angle_deg <= 270:
                dir_id = '9'
            elif 270 < angle_deg <= 300:
                dir_id = '10'
            elif 300 < angle_deg <= 330:
                dir_id = '11'
            else:
                dir_id = '0'
            
            if dir_id in image_dict:
                waypoint_images[dir_id] = image_dict[dir_id]
                waypoint_distances[dir_id] = batch_distance[angle_idx]
                waypoint_radius[dir_id] = angles[angle_idx]
                
        return waypoint_images, waypoint_radius, waypoint_distances

    def _eval_vlm(self) -> None:
        """
        Evaluation using VLM-based navigation.
        
        This method is the main evaluation loop that:
        1. Initializes the VLM navigator
        2. For each episode, uses VLM to make navigation decisions
        3. Records metrics and generates visualizations
        """
        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            try:
                graphs_file = config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE
            except Exception:
                graphs_file = None

            use_vlnce_map = False
            if graphs_file:
                graph_path = graphs_file if os.path.isabs(graphs_file) else os.path.join(os.getcwd(), graphs_file)
                if os.path.exists(graph_path):
                    use_vlnce_map = True

            if use_vlnce_map:
                if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            else:
                if "TOP_DOWN_MAP" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")

            if "COLLISIONS" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        else:
            config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}_vlm.json",
            )
            if os.path.exists(fname):
                print(f"VLM Evaluation exists at: {fname}")
                print("Overwriting previous results...")

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        )

        dataset_length = sum(envs.number_of_episodes)
        print(f'[VLM Trainer] local rank: {self.local_rank} | dataset length: {dataset_length}')

        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()
        observations = envs.reset()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # Set up the VLM logger
        log_file = "./navigator_vlm_log.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_vlm_logger")
        nav_logger.addHandler(logging.FileHandler(filename=log_file))
        
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        # Cache for action/landmark extraction (reusable across VLM and LLM)
        actions_cache_path = f"./cache_files/{dataset_name}/actions_cache_vlm.json"
        if os.path.exists(actions_cache_path):
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {}
        
        # Initialize VLM Navigator
        vlm_port = getattr(config, 'VLM_PORT', 23333)
        vlm_type = getattr(config, 'VLM', 'Qwen2-VL-72B')
        navigator = Open_Nav_VLM(
            self.device, 
            vlm_type=vlm_type,
            vlm_port=vlm_port,
            use_global_reflector=True  # Enable Global Reflector with UARF
        )
        logger.info(f"[Navigator] Initialized with VLM={vlm_type}, GlobalReflector={navigator.use_global_reflector}, VGM={navigator.use_vgm}")
        
        current_step = 0
        nav_history = []
        error_number = 0
        
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []
            headings = []
            
            for ob_i in range(len(current_episodes)):
                agent_state_i = envs.call_at(ob_i, "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            
            # ========== VLM Navigator Start ==========
            episode_id = current_episodes[0].episode_id
            nav_logger.info(f"==================== Episode {episode_id} ====================")
            nav_logger.info(f"Instruction: {instruction}")
            
            # Extract actions and landmarks (cached for efficiency)
            actions, landmarks = "", ""
            if instruction not in actions_cache:
                nav_logger.info("Extracting actions and landmarks...")
                actions = navigator.get_actions(instruction)
                landmarks = navigator.get_landmarks(actions)
                actions_cache[instruction] = {"actions": actions, "landmarks": landmarks}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
            else:
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction]["landmarks"]
                
            nav_logger.info(f"Actions: {actions}")
            nav_logger.info(f"Landmarks: {landmarks}")
            
            # Determine step limit based on action complexity
            # step_length = 6 if len(actions.split("\n")) <= 6 else 8
            step_length = 6 if len(actions.split("\n")) <= 6 else 8

            stop_flag = False
            current_step += 1
            nav_logger.info(f"-------------------- Step {current_step} --------------------")
            
            with torch.no_grad():
                # Candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net(
                    mode="waypoint",
                    waypoint_predictor=self.waypoint_predictor,
                    observations=batch,
                    in_train=False,
                )
            
            # Construct image dictionaries
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(
                batch_distances[-1], batch_angles, images_list
            )
            
            # ========== VLM Observation Processing ==========
            nav_logger.info("========== VLM Observation Processing ==========")
            observation_results, observe_dict, images_for_vlm, spatial_descriptions = navigator.observe_environment_vlm(
                nav_logger, current_step, images_dict, 
                episode_id=episode_id, instruction=instruction,
                landmarks=landmarks  # Pass landmarks for VGM node tagging
            )
            observation = "; ".join(observation_results)
            
            nav_logger.info("========== Review History ==========")
            history_traj = navigator.review_history(nav_logger, nav_history) if len(nav_history) > 0 else "Step 0 start position."

            if not stop_flag:
                nav_logger.info("========== Estimate Completion Progress ==========")
                estimation = navigator.estimate_completion(
                    nav_logger, actions, landmarks, history_traj, current_step
                )
                
                nav_logger.info("========== VLM Navigation Decision ==========")
                # Get available candidate viewpoints
                try:
                    candidate_keys = list(radius_dict.keys())
                except Exception:
                    candidate_keys = []

                # VLM multi-modal navigation decision
                predictions, thoughts, break_flag = navigator.move_to_next_vp_vlm(
                    nav_logger, current_step, instruction, actions, landmarks,
                    history_traj, estimation, observation, observe_dict,
                    images_for_vlm, spatial_descriptions, candidate_keys
                )

                nav_logger.info("========== Thought Fusion ==========")
                fused_pred_thought = navigator.thought_fusion(nav_logger, predictions, thoughts)
                
                nav_logger.info("========== VLM Decision Test ==========")
                # Get valid candidate viewpoints
                try:
                    available_keys = set(radius_dict.keys())
                except Exception:
                    available_keys = set()

                next_vp, thought, error_number = navigator.test_decisions_vlm(
                    nav_logger, fused_pred_thought, observation, instruction,
                    error_number, observe_dict, available_keys,
                    images_for_vlm, spatial_descriptions
                )
           
            try:
                if not stop_flag:
                    env_actions = []
                    env_actions.append({
                        'action': {
                            'action': 4,
                            'action_args': {
                                'angle': radius_dict[next_vp],
                                'distance': distance_dict[next_vp],
                            }
                        }
                    })
                    nav_logger.info(f"Final env action: {env_actions}")
                    outputs = envs.step(env_actions)
                    
                    curr_observe = observe_dict.get(next_vp, f"Direction {next_vp}")
                    nav_logger.info("========== Save History ==========")
                    nav_history = navigator.save_history(
                        nav_logger, current_step, next_vp, thought, curr_observe, nav_history,
                        spatial_desc=spatial_descriptions.get(next_vp, "")
                    )
                
                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                    
                    # Render frame for video
                    try:
                        for i in range(envs.num_envs):
                            frame = observations_to_image(observations[i], infos[i])
                            if config.VIDEO_RENDER_TEXT_INFO:
                                frame = append_text_to_image(frame, instruction)
                            rgb_frames[i].append(frame)
                    except Exception as _e:
                        pass
                    
                    instruction, images_list = self.generate_input(observations[-1])
                    error_number = 0
                    
                    # Check if navigation should stop
                    if current_step == step_length:
                        dones[0] = True
                    else:
                        for j, ob in enumerate(observations):
                            envs.call_at(j,
                                'change_current_path',
                                {'new_path': ob.pop('positions'),
                                 'collisions': ob.pop('collisions')}
                            )
                else:
                    dones[0] = True
                
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device
                )
                
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue
                    
                    # Reset for new episode
                    current_step = 0
                    nav_history = []
                    navigator.reset_episode()  # Reset VLM navigator state
                    
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                    
                    if 'current_path' in envs.current_episodes()[i].info.keys():
                        positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                        collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                        collisions_ = np.array([])
                        
                    distance = np.array(info['position']['distance']).astype(float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1], axis=1).sum()
                    metric['collisions'] = collisions_.mean() if len(collisions_) > 0 else 0.0
                    gt_length = distance[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])

                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW
                    stats_episodes[current_episodes[i].episode_id] = metric

                    # Save video if enabled
                    if len(config.VIDEO_OPTION) > 0 and len(rgb_frames[i]) > 0:
                        try:
                            try:
                                desired_fps = int(os.environ.get("VIDEO_FPS", "5"))
                            except Exception:
                                desired_fps = 5

                            frames = list(rgb_frames[i])
                            try:
                                frame_repeat = int(os.environ.get("FRAME_REPEAT", "4"))
                            except Exception:
                                frame_repeat = 4
                            if frame_repeat > 1 and len(frames) > 0:
                                repeated = []
                                for _im in frames:
                                    repeated.extend([_im] * frame_repeat)
                                frames = repeated

                            video_name = f"vlm_episode={current_episodes[i].episode_id}-spl={metric['spl']:.2f}"
                            images_to_video(frames, config.VIDEO_DIR, video_name, fps=desired_fps)
                        except Exception:
                            pass

                        # Dump trajectory JSON
                        try:
                            base_dir = config.VIDEO_DIR
                            traj_dir = os.path.join(os.path.dirname(base_dir), 'trajectories_vlm') if os.path.basename(base_dir) == 'videos' else os.path.join(base_dir, 'trajectories_vlm')
                            os.makedirs(traj_dir, exist_ok=True)
                            traj_payload = {
                                "episode_id": current_episodes[i].episode_id,
                                "positions": positions_.tolist(),
                                "gt_path": gt_path.tolist(),
                                "collisions": collisions_.tolist() if len(collisions_) > 0 else [],
                                "metrics": metric,
                            }
                            with open(os.path.join(traj_dir, f"vlm_episode_{current_episodes[i].episode_id}.json"), 'w') as tfp:
                                json.dump(traj_payload, tfp, indent=2)
                        except Exception:
                            pass

                        rgb_frames[i] = []

                    observations[i] = envs.reset_at(i)[0]
                    instruction, images_list = self.generate_input(observations[i])
                    
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                        
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)
                
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                    rgb_frames,
                )
                headings = headings.tolist()
                
            except Exception as e:
                nav_logger.info(f"Error in VLM navigation step: {e}")
                import traceback
                nav_logger.info(traceback.format_exc())
                current_step -= 1
                
        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
            
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total, dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k, v in aggregated_stats.items():
                v = torch.tensor(v * num_episodes).cuda()
                cat_v = gather_list_and_concat(v, self.world_size)
                v = (sum(cat_v) / total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_vlm_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}_vlm.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"[VLM] Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"[VLM] Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        """Collect validation trajectories."""
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        return trajectories
        
    def eval(self) -> None:
        """
        Main method of trainer evaluation using VLM.
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION', 'STEPS_TAKEN']
        
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # Camera setup
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_vlm()