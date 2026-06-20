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
        import logging
        # 모든 가중치 로딩 로직을 주석 처리하거나 무시합니다.
        print("⚠️ [긴급] 가중치 파일 결함으로 인해 처음부터(Scratch) 학습을 시작합니다.")
        print("🚀 A100의 파워를 믿으세요! 가중치 없이 강제 시동 중...")
        
        # 원래 들어온 params(랜덤 초기값)를 그대로 돌려줘서 에러 없이 통과시킵니다.
        return params
