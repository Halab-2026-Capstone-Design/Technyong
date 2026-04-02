"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import logging
import time
import os
import json
import dataclasses

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]

# =====================================================================
# 🌟 Comet 팀 기술 이식: Key Mismatch 해결을 위한 Transform (Problem 2 해결)
# =====================================================================
class SemanticKeyMapper(_transforms.DataTransformFn):
    """
    RFT 학습 시 저장된 원본 이미지 데이터를 모델이 요구하는 
    '_semantic' 키 이름으로 복사하여 KeyError를 방지합니다.
    """
    def __call__(self, data: dict) -> dict:
        if "observation/egocentric_camera" in data:
            data["observation/egocentric_camera_semantic"] = data["observation/egocentric_camera"]
            data["observation/wrist_image_left_semantic"] = data["observation/wrist_image_left"]
            data["observation/wrist_image_right_semantic"] = data["observation/wrist_image_right"]
        return data


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
    
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for BehaviorLeRobotDataset: {seed}")
        
    tasks = [
        "picking_up_trash", "putting_away_Halloween_decorations", "cleaning_up_plates_and_food", 
        "setting_mousetraps", "hiding_Easter_eggs", "set_up_a_coffee_station_in_your_kitchen", 
        "putting_dishes_away_after_cleaning", "preparing_lunch_box", "loading_the_car", 
        "carrying_in_groceries", "turning_on_radio", "picking_up_toys", "can_meat", 
        "rearranging_kitchen_furniture", "putting_up_Christmas_decorations_inside", 
        "bringing_in_wood", "moving_boxes_to_storage", "bringing_water", "tidying_bedroom", 
        "outfit_a_basic_toolbox", "sorting_vegetables", "collecting_childrens_toys",
        "putting_shoes_on_rack", "boxing_books_up_for_storage", "storing_food",
        "clearing_food_from_table_into_fridge", "assembling_gift_baskets", "sorting_household_items",
        "getting_organized_for_work", "clean_up_your_desk", "setting_the_fire", "clean_boxing_gloves",
        "wash_a_baseball_cap", "wash_dog_toys", "hanging_pictures", "attach_a_camera_to_a_tripod",
        "clean_a_patio", "clean_a_trumpet", "spraying_for_bugs", "spraying_fruit_trees",
        "make_microwave_popcorn", "cook_cabbage", "make_pizza", "chop_an_onion",
        "slicing_vegetables", "chopping_wood", "canning_food", "cook_hot_dogs",
        "cook_bacon", "freeze_pies"
    ]

    # =====================================================================
    # 🌟 Comet 팀 기술 이식: RFT 성공 에피소드 필터링 (Problem 3 해결)
    # =====================================================================
    episodes_to_load = data_config.episodes_index
    success_list_path = "success_list.jsonl"
    
    if os.path.exists(success_list_path):
        rft_episodes = []
        try:
            with open(success_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    # "run_dir 1001" 형태로 저장된 문자열 파싱
                    if " " in line and not line.startswith("{"):
                        _, eid_str = line.rsplit(" ", 1)
                        rft_episodes.append(int(eid_str))
                    # JSON 형태로 저장된 경우
                    elif line.startswith("{"):
                        data = json.loads(line)
                        rft_episodes.append(int(data.get("episode_id", 0)))
                        
            if rft_episodes:
                episodes_to_load = rft_episodes
                logging.info(f"🎯 RFT 모드 발동: success_list.jsonl에서 {len(rft_episodes)}개의 성공 에피소드만 골라서 학습합니다!")
        except Exception as e:
            logging.warning(f"⚠️ success_list.jsonl 파싱 실패 (전체 데이터 로드 진행): {e}")

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=tasks,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=episodes_to_load, # 🌟 필터링된 에피소드 ID 주입
        chunk_streaming_using_keyframe=False,
        shuffle=True,
        seed=seed,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # 🌟 SemanticKeyMapper 변환기를 가장 앞단에 추가하여 모델 입력을 맞춥니다.
    transforms_list = [
        *data_config.repack_transforms.inputs,
        SemanticKeyMapper(), # <--- Problem 2 해결
        *data_config.data_transforms.inputs,
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm 
        ),
    ]
    
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        if hasattr(transform, '__class__') and transform.__class__.__name__ == 'ComputeSubtaskStateFromMeta':
            from b1k import transforms as b1k_transforms
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                model_transforms.append(b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset))
            else:
                logging.warning("Skipping subtask state computation - dataset has no meta.episodes")
        else:
            model_transforms.append(transform)
    
    transforms_list.extend(model_transforms)

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    if not hasattr(dataset, 'episode_data_index'):
        raise ValueError("Dataset must have episode_data_index attribute")
    
    episode_data_index = dataset.episode_data_index
    episode_to = episode_data_index['to'] 
    episode_from = episode_data_index['from']
    episodes = dataset.episodes
    
    episode_lengths = {}
    for i, episode_index in enumerate(episodes):
        if i < len(episode_to) and i < len(episode_from):
            episode_length = episode_to[i] - episode_from[i]
            episode_lengths[episode_index] = float(episode_length)
    return episode_lengths


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    import jax
    import time
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
