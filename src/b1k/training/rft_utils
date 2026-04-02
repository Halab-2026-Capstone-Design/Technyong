import json
from pathlib import Path

def check_success_condition(run_dir: Path) -> bool:
    """해당 롤아웃 폴더가 성공한 에피소드인지 확인 (성공 시에만 .npz 파일이 저장됨)"""
    npz_files = list(run_dir.glob("*.npz"))
    return len(npz_files) > 0

def create_episode_id(task_name: str, task_map: dict, success_count: int) -> int:
    """태스크별 고유 에피소드 ID 생성"""
    base_id = task_map.get(task_name, 0)
    return base_id + success_count

def update_success_list(list_path: Path, rel_path: str, episode_id: int):
    """성공 목록 jsonl 파일에 안전하게 이어쓰기(Append)"""
    with open(list_path, "a") as f:
        data = {"path": rel_path, "episode_id": episode_id}
        f.write(json.dumps(data) + "\n")
