import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 데이터셋 episode_index 기준: task_idx * 10000 + 10, 증가 단위 10
# 예) turning_on_radio(idx=0): 10, 20, 30 ...
#     picking_up_trash(idx=1): 10010, 10020, 10030 ...
TASK_EPISODE_BASE = {
    "turning_on_radio":        0 * 10000 + 10,
    "picking_up_trash":        1 * 10000 + 10,
    "bringing_water":         17 * 10000 + 10,
    "tidying_bedroom":        18 * 10000 + 10,
    "putting_shoes_on_rack":  22 * 10000 + 10,
}
EPISODE_STEP = 10


def check_success_from_metrics(metrics_json_path: Path, threshold: float = 1.0) -> bool:
    """eval.py가 저장한 metrics JSON에서 q_score >= threshold 인지 확인."""
    try:
        with open(metrics_json_path) as f:
            data = json.load(f)
        q_score = data.get("q_score", {}).get("final", 0.0)
        return float(q_score) >= threshold
    except Exception as e:
        logger.warning(f"metrics 파싱 실패 ({metrics_json_path}): {e}")
        return False


def create_episode_id(task_name: str, success_count: int) -> int:
    """성공 순번 → 데이터셋 episode_index 매핑."""
    base = TASK_EPISODE_BASE.get(task_name)
    if base is None:
        raise ValueError(f"TASK_EPISODE_BASE에 없는 task: {task_name}")
    return base + success_count * EPISODE_STEP


def update_success_list(list_path: Path, metrics_filename: str, episode_id: int):
    """성공 목록 jsonl 파일에 이어쓰기."""
    with open(list_path, "a") as f:
        f.write(json.dumps({"metrics_file": metrics_filename, "episode_id": episode_id}) + "\n")
