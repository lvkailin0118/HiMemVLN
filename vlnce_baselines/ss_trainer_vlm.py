"""
VLM-based Schedule Sampler Trainer
Registers the VLM trainer with baseline_registry for use with Open-Nav VLM.
"""
import warnings
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.common.base_il_trainer_vlm import BaseVLNCETrainerVLM

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


@baseline_registry.register_trainer(name="schedulesampler-OPENNAV-VLM")
class SSTrainerVLM(BaseVLNCETrainerVLM):
    """
    VLM-based trainer for Open-Nav navigation.
    
    Uses Qwen2-VL (or similar VLM) for direct visual understanding,
    replacing the traditional LLM + RAM text description approach.
    """
    
    def __init__(self, config=None):
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()







