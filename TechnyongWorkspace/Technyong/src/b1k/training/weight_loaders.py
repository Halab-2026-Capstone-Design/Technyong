"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        import openpi.models.model as _model
        import numpy as np

        params_path = download.maybe_download(self.params_path)

        try:
            loaded_params = _model.restore_params(params_path, restore_type=np.ndarray)

            has_task_embeddings = 'task_embeddings' in loaded_params

            if has_task_embeddings:
                logger.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
                return _merge_params(loaded_params, params, missing_regex="^$")
            else:
                logger.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")
                missing_regex = (
                    ".*task_embeddings.*|"
                    ".*stage_pred_from_vlm.*|"
                    ".*task_stage_embeddings.*|"
                    ".*gate_sincos.*|"
                    ".*gate_task_stage.*|"
                    ".*gate_task.*|"
                    ".*fusion_layer.*|"
                    ".*stage_projection.*|"
                    ".*task_subtask_fusion.*|"
                    ".*fast_token_embedding.*|"
                    ".*fast_token_proj.*|"
                    ".*kv_transform.*"
                )
                return _merge_params(loaded_params, params, missing_regex=missing_regex)

        except Exception as e:
            logger.warning(f"Checkpoint loading failed ({e}). Starting from random init.")
            return params
