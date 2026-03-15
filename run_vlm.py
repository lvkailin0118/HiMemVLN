#!/usr/bin/env python3
"""
VLM-based Navigation Runner

This script runs the VLM-based navigation system using Qwen2-VL for
direct visual understanding instead of LLM + RAM text descriptions.

Usage:
    python run_vlm.py --exp_name test-vlm --exp-config run_OpenNav_VLM.yaml --vlm Qwen2-VL-72B --vlm_port 23333

Requirements:
    - vllm server running with Qwen2-VL model on specified port
    - SpatialBot model loaded for spatial understanding
"""
import os
import torch
import random
import argparse
import numpy as np
from habitat import logger
import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401

# Import VLM trainer to register it
from vlnce_baselines import ss_trainer_vlm  # noqa: F401

from vlnce_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry


def main():
    parser = argparse.ArgumentParser(description="Run VLM-based navigation")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test-vlm",
        required=True,
        help="Experiment ID for logging and results",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="Path to config yaml containing experiment info",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        '--local_rank', 
        type=int, 
        default=0, 
        help="Local GPU ID"
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="Qwen2-VL-72B",
        help="VLM model name (must match --served-model-name in vllm server)",
    )
    parser.add_argument(
        "--vlm_port",
        type=int,
        default=23333,
        help="Port for vllm server (default: 23333)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="not-needed",
        help="API key (not needed for local vllm deployment)",
    )
    
    args = parser.parse_args()
    
    # Allow torchrun to set the correct local rank for each process
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        try:
            args.local_rank = int(env_local_rank)
        except Exception:
            pass
    
    run_exp(**vars(args))


def run_exp(exp_name: str, exp_config: str,
            opts=None, local_rank=None,
            vlm: str = "Qwen2-VL-72B", 
            vlm_port: int = 23333,
            api_key: str = "not-needed",
            episodes_to_load: int = 0) -> None:
    """
    Runs VLM-based navigation experiment.
    
    Args:
        exp_name: Experiment name for logging
        exp_config: Path to config file
        opts: List of additional config options
        local_rank: GPU rank for distributed training
        vlm: VLM model name
        vlm_port: vllm server port
        api_key: API key (not needed for local deployment)
        episodes_to_load: Number of episodes to load (0 = all)
    """
    config = get_config(exp_config, opts)
    config.defrost()

    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.LOG_FILE = exp_name + '_vlm_' + config.LOG_FILE

    config.TASK_CONFIG.SEED = 0

    # Sync local_rank/world_size with torchrun if present
    try:
        env_ws = os.environ.get("WORLD_SIZE")
        env_lr = os.environ.get("LOCAL_RANK")
        if env_ws is not None:
            config.GPU_NUMBERS = int(env_ws)
        if env_lr is not None:
            config.local_rank = int(env_lr)
        else:
            config.local_rank = local_rank
    except Exception:
        config.local_rank = local_rank

    # VLM-specific configuration
    config.VLM = vlm
    config.VLM_PORT = vlm_port
    config.API_KEY = api_key
    
    # For compatibility with LLM code paths
    config.LLM = vlm

    config.freeze()
    
    # Check if the logs directory exists; if not, create it
    log_dir = 'logs/running_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add the file handler for logging
    logger.add_filehandler(os.path.join(log_dir, config.LOG_FILE))

    # Set random seeds for reproducibility
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # Get and initialize the VLM trainer
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    
    print(f"[VLM Runner] Using trainer: {config.TRAINER_NAME}")
    print(f"[VLM Runner] VLM model: {vlm}")
    print(f"[VLM Runner] VLM port: {vlm_port}")
    
    trainer = trainer_init(config)
    trainer.eval()


if __name__ == "__main__":
    __spec__ = None
    main()







