"""Training configuration for BEHAVIOR-1K challenge.

Reference: https://github.com/Physical-Intelligence/openpi
"""
import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, List, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms
import openpi.shared.download as _download

# Import from B1K custom modules
from b1k.models import pi_behavior_config
from b1k.policies import b1k_policy
from b1k.shared import normalize as _normalize
from b1k.training import weight_loaders
from b1k import transforms as b1k_transforms

# =========================================================
# (중략: 1등 팀의 DataConfig, TrainConfig 등 클래스 정의 부분)
# 1등 팀 원본에 있던 클래스들을 여기에 그대로 둡니다.
# =========================================================
ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None

@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    use_per_timestamp_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    rlds_data_dir: str | None = None
    behavior_dataset_root: str | None = None
    episodes_index: List[int] | None = None

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group: ...

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                b1k_transforms.ComputeSubtaskStateFromMeta(dataset=None),
                b1k_transforms.TaskIndexToTaskId(),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig: ...
    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,
        )
    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None: return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            return norm_stats
        except FileNotFoundError:
            return None

@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)
    use_delta_joint_actions: bool = False
    use_fast_tokenization: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_mapping = {
            "observation/egocentric_camera": "observation.images.rgb.head",
            "observation/wrist_image_left": "observation.images.rgb.left_wrist",
            "observation/wrist_image_right": "observation.images.rgb.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "task_index": "task_index",
            "timestamp": "timestamp",
            "episode_index": "episode_index",
            "index": "index",
        }
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_mapping)])
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            base_config = self.create_base_config(assets_dirs, model_config)
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "B1K"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_behavior_config.PiBehaviorConfig)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotB1KDataConfig)
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int | None = None
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1
    val_log_interval: int = 100
    val_batch_size: int | None = None
    val_num_batches: int = 10
    val_repo_id: str | None = None
    val_episodes_index: List[int] | None = None
    num_flow_samples: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name: raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

# =========================================================
# 💡 핵심 파트: B1K Training Configurations
# =========================================================
_CONFIGS = [
    # 1. 1등 팀의 원본 SFT 설정 (베이스라인)
    TrainConfig(
        name="pi_behavior_b1k_fast",
        exp_name="openpi",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30, action_dim=32, use_correlated_noise=True, correlation_beta=0.5,
            use_fast_auxiliary=True, fast_loss_weight=0.05, fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024, max_fast_tokens=200, use_kv_transform=True,
            use_knowledge_insulation=False, subtask_loss_weight=0.1, freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False, behavior_dataset_root="~/data/behavior_224_rgb", use_per_timestamp_norm=True,
            ),
            use_delta_joint_actions=True, use_fast_tokenization=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=1e-4, decay_steps=20_000, decay_lr=1e-5),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000, assets_base_dir="./outputs/assets", checkpoint_base_dir="./outputs/checkpoints",
        num_workers=80, save_interval=500, keep_period=2000,
    ),
    
    # 🌟 2. Comet 팀 기술 이식: RFT 전용 경량화 설정 (우리가 새로 추가한 부분)
    TrainConfig(
        name="pi_behavior_b1k_fast_rft",
        exp_name="rft_round",
        project_name="B1K_RFT",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30, action_dim=32,
            use_fast_auxiliary=True,
            freeze_vision_backbone=True,  # RFT는 비전은 얼리고 팔만 미세조정
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(use_per_timestamp_norm=True),
            use_delta_joint_actions=True, use_fast_tokenization=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=2e-5, decay_steps=2000, decay_lr=1e-6
        ), # 짧고 약하게 학습
        
        # 이전 라운드의 체크포인트를 자동으로 물고 들어옴 (파이프라인 연동)
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            os.environ.get("RFT_CKPT_PATH", "gs://openpi-assets/checkpoints/pi05_base/params")
        ),
        num_train_steps=2000, # 적은 데이터에 맞게 스텝 최소화
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=16, save_interval=500,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
