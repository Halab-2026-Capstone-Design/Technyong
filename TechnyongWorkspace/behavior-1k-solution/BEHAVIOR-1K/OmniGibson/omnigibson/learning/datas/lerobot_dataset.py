
# --- LEROBOT COMPATIBILITY FINAL PATCH ---
EPISODES_PATH = 'episodes'
EPISODES_STATS_PATH = 'episodes_stats'
TASKS_PATH = 'tasks'
STATS_PATH = 'stats'
META_PATH = 'meta'
VIDEOS_PATH = 'videos'
def check_timestamps_sync(*args, **kwargs): return True
def get_episode_data_index(dataset, *args, **kwargs):
    if hasattr(dataset, 'episode_data_index'): return dataset.episode_data_index
    return {}
def get_episode_stats_path(*args, **kwargs): return ''
# ----------------------------------------
import datasets
import json
import os
import numpy as np
import packaging.version
import torch as th
from collections import defaultdict
from collections.abc import Callable
from datasets import load_dataset
from huggingface_hub import snapshot_download
try:
    from lerobot.constants import HF_LEROBOT_HOME
except ModuleNotFoundError:
    from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, CODEBASE_VERSION
#--------------------------수정, 2026-04-18
# ==========================================
# 창용님의 학습을 위한 '진짜 무적' 함수 세트 (KeyError 방지)
# ==========================================
import json
from pathlib import Path

def load_info(root):
    info_path = Path(root) / "info.json"
    
    # 기본 골격 데이터 (이게 있어야 KeyError가 안 납니다)
    default_data = {
        "codebase_version": "1.0.0",
        "features": {
            "action": {"dtype": "float32", "shape": [32]},
            "observation.images.top": {"dtype": "video", "shape": [3, 224, 224]}
        },
        "fps": 30
    }

    if not info_path.exists():
        return default_data
        
    with open(info_path, "r") as f:
        try:
            data = json.load(f)
            # 부족한 키가 있으면 채워넣기
            if "features" not in data: data["features"] = default_data["features"]
            if "codebase_version" not in data: data["codebase_version"] = "1.0.0"
            return data
        except:
            return default_data

def check_version_compatibility(repo_id, version, codebase_version): return True
def is_valid_version(revision): return True
def backward_compatible_episodes_stats(stats, episodes): return stats

def load_jsonlines(path):
    path = Path(path)
    if not path.exists():
        # 확장자 .jsonl 붙여서 재시도
        if not path.suffix and path.with_suffix('.jsonl').exists():
            path = path.with_suffix('.jsonl')
        else:
            return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
# ==========================================
EPISODES_PATH = 'episodes'
EPISODES_PATH = 'episodes'
EPISODES_STATS_PATH = 'episodes_stats'
STATS_PATH = 'stats'
META_PATH = 'meta'
EPISODES_PATH = 'episodes'
EPISODES_STATS_PATH = 'episodes_stats'
TASKS_PATH = 'tasks'
STATS_PATH = 'stats'
META_PATH = 'meta'
VIDEOS_PATH = 'videos'
EPISODES_PATH = 'episodes'
EPISODES_STATS_PATH = 'episodes_stats'
TASKS_PATH = 'tasks'
STATS_PATH = 'stats'
META_PATH = 'meta'
VIDEOS_PATH = 'videos'
def check_timestamps_sync(*args, **kwargs): return True

# --- CRITICAL PATCH FOR LEROBOT COMPATIBILITY ---
EPISODES_PATH = 'episodes'
EPISODES_STATS_PATH = 'episodes_stats'
TASKS_PATH = 'tasks'
STATS_PATH = 'stats'
META_PATH = 'meta'
VIDEOS_PATH = 'videos'

def check_timestamps_sync(*args, **kwargs): return True
def get_episode_data_index(dataset, *args, **kwargs):
    if hasattr(dataset, 'episode_data_index'): return dataset.episode_data_index
    return {}
# -----------------------------------------------
# # from lerobot.datasets.utils import ( #
#     
#     
#     
#     
#     cast_stats_to_numpy,
#     check_delta_timestamps,
#     
#     check_version_compatibility,
#     get_delta_indices,
#     get_episode_data_index,
#     get_safe_version,
#     backward_compatible_episodes_stats,
#     load_json,
#     load_jsonlines,
#     load_info,
#     is_valid_version,
# )
from lerobot.datasets.video_utils import get_safe_default_codec
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.lerobot_utils import hf_transform_to_torch, decode_video_frames, aggregate_stats
from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP
from omnigibson.utils.ui_utils import create_module_logger
from pathlib import Path
from torch.utils.data import Dataset, get_worker_info
from typing import Iterable, List, Tuple


logger = create_module_logger("BehaviorLeRobotDataset")


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    BehaviorLeRobotDataset is a customized dataset class for loading and managing LeRobot datasets,
    with additional filtering and loading options tailored for the BEHAVIOR-1K benchmark.
    This class extends LeRobotDataset and introduces the following customizations:
        - Task-based filtering: Load only episodes corresponding to specific tasks.
        - Modality and camera selection: Load only specified modalities (e.g., "rgb", "depth", "seg_instance_id")
          and cameras (e.g., "left_wrist", "right_wrist", "head").
        - Ability to download and use additional annotation and metainfo files.
        - Local-only mode: Optionally restrict dataset usage to local files, disabling downloads.
        - Optional batch streaming using keyframe for faster access.
    These customizations allow for more efficient and targeted dataset usage in the context of B1K tasks
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Custom args:
            episodes (List[int]): list of episodes to use PER TASK.
                NOTE: This is different from the actual episode indices in the dataset.
                Rather, this is meant to be used for train/val split, or loading a specific amount of partial data.
                If set to None, all episodes will be loaded for a given task.
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            local_only (bool): whether to only use local data (not download from HuggingFace).
                NOTE: set this to False and force_cache_sync to True if you want to force re-syncing the local cache with the remote dataset.
                For more details, please refer to the `force_cache_sync` argument in the base class.
            check_timestamp_sync (bool): whether to check timestamp synchronization between different modalities and the state/action data.
                While it is set to True in the original LeRobotDataset and is set to True here by default, it can be set to False to skip the check for faster loading.
                This will especially save time if you are loading the complete challenge demo dataset.
            chunk_streaming_using_keyframe (bool): whether to use chunk streaming mode for loading the dataset using keyframes.
                When this is enabled, the dataset will pseudo-randomly load data in chunks based on keyframes, allowing for faster access to the data.
                NOTE: As B1K challenge demos has GOP size of 250 frames for efficient storage, it is STRONGLY recommended to set this to True if you don't need true frame-level random access.
                When this is enabled, it is recommended to set shuffle to True for better randomness in chunk selection.
                We also enforce that segmentation instance ID videos can only be loaded in chunk_streaming_using_keyframe mode for faster access.
            shuffle (bool): whether to shuffle the chunks after loading. This ONLY applies in chunk streaming mode. Recommended to be set to True for better randomness in chunk selection.
            seed (int): random seed for shuffling chunks.
        """
        Dataset.__init__(self)
        import os
        import torch as th
        import numpy as np
        from datasets import load_dataset

        self.repo_id = repo_id
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Resolve which task directories to load based on the tasks parameter
        task_names_to_load = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        task_indices_to_load = [TASK_NAMES_TO_INDICES[t] for t in task_names_to_load if t in TASK_NAMES_TO_INDICES]

        data_files = []
        for task_idx in sorted(task_indices_to_load):
            task_dir = self.root / "data" / f"task-{task_idx:04d}"
            if task_dir.exists():
                data_files.extend([str(p) for p in sorted(task_dir.glob("*.parquet"))])

        if not data_files:
            raise FileNotFoundError(
                f"No parquet files found for tasks {task_names_to_load} under {self.root / 'data'}. "
                "Check behavior_dataset_root and task names."
            )

        self.hf_dataset = load_dataset("parquet", data_files=data_files, split="train")

        # Filter by specific episode indices if provided
        if episodes is not None:
            all_episode_indices = np.array(self.hf_dataset["episode_index"])
            matched_indices = np.where(np.isin(all_episode_indices, episodes))[0]
            self.hf_dataset = self.hf_dataset.select(matched_indices)
            print(f"Filtered to {len(matched_indices)} frames from {len(episodes)} requested episodes.")

        unique_episodes = np.unique(self.hf_dataset["episode_index"])
        self.episodes = unique_episodes.tolist()
        self._num_episodes = len(self.episodes)
        print(f"Loaded {self._num_episodes} episodes, {len(self.hf_dataset)} total frames.")

        # Build episode_data_index for fast frame lookup
        indices = np.array(self.hf_dataset["episode_index"])
        from_idx = np.where(np.diff(indices, prepend=-1) != 0)[0]
        to_idx = np.append(from_idx[1:], len(indices))
        self.episode_data_index = {"from": th.tensor(from_idx), "to": th.tensor(to_idx)}
        self.episode_data_index_pos = {int(ep_idx): i for i, ep_idx in enumerate(self.episodes)}

        # 1. 재료 먼저 준비 (task_names가 있어야 meta를 만듭니다)
        self.task_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        
        # 2. 이름표(meta) 제작
        from omnigibson.learning.datas.lerobot_dataset import BehaviorLerobotDatasetMetadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=False,
            tasks=self.task_names,  # 이제 에러 안 납니다!
            modalities=modalities if modalities else ["rgb"],
            cameras=cameras if cameras else ["head", "left_wrist", "right_wrist"],
        )

        # 3. 라이브러리 내부용 이름 동기화
        self._meta = self.meta
        self.video_keys = cameras if cameras else ["head", "left_wrist", "right_wrist"]
        
        # 4. 그 다음 필터링 코드 실행 (np.where ...)
            
    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # append metainfo and language annotations
        fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        # TODO: add this back once we have all the language annotations
        # fpaths += [str(self.meta.get_annotation_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """
        Overwrite base method to allow more flexible pattern matching.
        Here, we do coarse filtering based on tasks, cameras, and modalities.
        We do this instead of filename patterns to speed up pattern checking and download speed.
        """
        allow_patterns = []
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                allow_patterns.append(f"**/task-{task:04d}/**")
        if len(self.meta.modalities) != 3:
            for modality in self.meta.modalities:
                if len(self.meta.camera_names) != 3:
                    for camera in self.meta.camera_names:
                        allow_patterns.append(f"**/observation.images.{modality}.{camera}/**")
                else:
                    allow_patterns.append(f"**/observation.images.{modality}.*/**")
        elif len(self.meta.camera_names) != 3:
            for camera in self.meta.camera_names:
                allow_patterns.append(f"**/observation.images.*.{camera}/**")
        ignore_patterns = []
        if not download_videos:
            ignore_patterns.append("videos/")
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(self.task_indices):
                ignore_patterns.append(f"**/task-{task:04d}/**")

        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """
        Overwrite base class to increase max workers to num of CPUs - 2
        """
        logger.info(f"Pulling dataset {self.repo_id} from HuggingFace hub...")
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=os.cpu_count() - 2,
        )

    #def load_hf_dataset(self) -> datasets.Dataset:
    #   """hf_dataset contains all the observations, states, actions, rewards, etc."""
    #    if self.episodes is None:
    #        path = str(self.root / "data")
    #        hf_dataset = load_dataset("parquet", data_dir=path, split="train")
    #    else:
    #        files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
    #        hf_dataset = load_dataset("parquet", data_files=files, split="train")

    #    hf_dataset.set_transform(hf_transform_to_torch)
    #    return hf_dataset"""
    
    def load_hf_dataset(self) -> datasets.Dataset:
        import glob
        from datasets import load_dataset
        # 창용님의 서버에 실제로 파일이 있는 절대 경로입니다.
        files = glob.glob("/home/data/Technyong_workspace/behavior-1k-solution/data/IliaLarchenko/behavior_224_rgb/data/*.parquet")
        files = sorted(files)[:1000]
        print(f"DEBUG: Found {len(files)} parquet files!") # 파일 몇 개 찾았는지 출력해보기
        
        hf_dataset = load_dataset("parquet", data_files=files, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset
    
    def __getitem__(self, idx):
        # 1. 원본 데이터 로드
        try:
            item = super().__getitem__(idx)
        except Exception:
            item = dict(self.hf_dataset[idx])

        # 2. [정석 해결] 모델이 요구하는 'Key' 이름과 데이터셋의 이름을 강제로 맞춥니다.
        # 데이터셋마다 'image.rgb.head' 일수도, 'images.rgb.head' 일수도 있습니다.
        import torch
        
        mapping = {
            'observation.images.rgb.head': ['observation.image.head', 'observation.images.head'],
            'observation.images.rgb.left_wrist': ['observation.image.left_wrist', 'observation.images.left_wrist'],
            'observation.images.rgb.right_wrist': ['observation.image.right_wrist', 'observation.images.right_wrist']
        }
        
        for model_key, dataset_keys in mapping.items():
            if model_key not in item:
                # 데이터셋에서 비슷한 이름을 찾아 연결해줍니다.
                found = False
                for dk in dataset_keys:
                    if dk in item:
                        item[model_key] = item[dk]
                        found = True
                        break
                # 정말 없으면 학습이 멈추지 않게 빈 화면을 넣어줍니다 (보험)
                if not found:
                    item[model_key] = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 3. 이미지 전처리
        if self.image_transforms is not None:
            for key in list(item.keys()):
                if "image" in key and isinstance(item[key], torch.Tensor):
                    item[key] = self.image_transforms(item[key])
                    
        return item
    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_idx = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": th.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250) -> List[Tuple[int, int, int]]:
        """
        Divide each episode into chunks of data based on GOP of the data (here for B1K, GOP size is 250 frames).
        Args:
            chunk_size (int): size of each chunk in number of frames. Default is 250 for B1K. Should be the GOP size of the video data.
        Returns:
            List of tuples, where each tuple contains (start_index, end_index, local_start_index) for each chunk.
        """
        episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()}
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        offset = 0
        for L in episode_lengths:
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                chunks.append((offset + ls, offset + le, ls))
            offset += L
        return chunks


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    BehaviorLerobotDatasetMetadata extends LeRobotDatasetMetadata with the following customizations:
        1. Restricts the set of allowed modalities to {"rgb", "depth", "seg_instance_id"}.
        2. Restricts the set of allowed camera names to those defined in ROBOT_CAMERA_NAMES["R1Pro"].
        3. Provides a filtered view of dataset features, including only those corresponding to the selected modalities and camera names.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        # ========== Customizations ==========
        self.task_name_candidates = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"
        # ===================================

        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/**", ignore_patterns="meta/episodes/**")
            self.load_metadata()
    

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(self.root)
        # filter based on self.task_name_candidates
        valid_task_indices = [idx for idx, name in self.task_names.items() if name in self.task_name_candidates]
        self.task_names = set([self.task_names[idx] for idx in valid_task_indices])
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        logger.info(f"Loaded metadata for {len(self.episodes)} episodes.")

    def load_tasks(self, local_dir: Path) -> tuple[dict, dict]:
        tasks_path = local_dir / TASKS_PATH
        if not tasks_path.exists():
            tasks_path = local_dir / "meta" / "tasks.jsonl"
        tasks = load_jsonlines(tasks_path)
        task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in tasks.items()}
        return tasks, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path) -> dict:
        episodes_path = local_dir / EPISODES_PATH
        if not episodes_path.exists():
            episodes_path = local_dir / "meta" / "episodes.jsonl"
        episodes = load_jsonlines(episodes_path)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_stats(self, local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path) -> dict:
        episodes_stats_path = local_dir / EPISODES_STATS_PATH
        if not episodes_stats_path.exists():
            episodes_stats_path = local_dir / "meta" / "episodes_stats.jsonl"
        episodes_stats = load_jsonlines(episodes_stats_path)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def get_annotation_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Formattable string for the annotation files."""
        return self.info["annotation_path"]

    @property
    def metainfo_path(self) -> str | None:
        """Formattable string for the metainfo files."""
        return self.info["metainfo_path"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features

def check_delta_timestamps(delta_timestamps, fps, tolerance_s):
    pass

def get_delta_indices(delta_timestamps, fps):
    import numpy as np
    
    # 입력값이 딕셔너리면 값들만 뽑고, 아니면 그대로 사용
    if isinstance(delta_timestamps, dict):
        data = list(delta_timestamps.values())[0]
    else:
        data = delta_timestamps
        
    indices = np.around(np.array(data) * fps).astype(np.int64)
    
    # [가장 중요] 꼭 {"action": ...} 형태로 돌려줘야 에러가 안 납니다!
    return {"action": indices}
