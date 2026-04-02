import argparse
import logging
import os
import re
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ================================================================
# [주석처리] 기존 단순 SFT 1회 실행 방식 — 이걸 RFT 루프로 교체
# ================================================================
# def run_once():
#     subprocess.run([
#         "uv", "run", "scripts/train.py",
#         "pi_behavior_b1k_fast",
#         "--num_train_steps=200000",
#         "--overwrite",
#     ], check=True)
# ================================================================


# Step 1: 롤아웃
def run_rollout(task_name: str, rollout_output_dir: str, num_instances: int):
    """
    현재 서버(serve_b1k.py)에 띄워진 정책으로 시뮬레이터 롤아웃 실행.
    성공/실패 결과가 rollout_output_dir 아래 각 폴더에 저장됨.
    """
    logger.info(f"[Step 1] 롤아웃 시작: task={task_name}, instances={num_instances}")

    Path(rollout_output_dir).mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "python", "OmniGibson/omnigibson/learning/eval_custom.py",
        "policy=websocket",
        "save_rollout=true",
        "perturb_pose=true",                              # 포즈 랜덤 변형 (RFT 핵심)
        f"task.name={task_name}",
        f"log_path={rollout_output_dir}",
        "use_parallel_evaluator=false",
        "parallel_evaluator_start_idx=0",
        f"parallel_evaluator_end_idx={num_instances}",
        "model.port=8000",
        "env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper",
    ], check=True)

    logger.info(f"[Step 1] 롤아웃 완료 → {rollout_output_dir}")


# Step 2: 성공 필터링
def filter_success_episodes(
    rollout_output_dir: str,
    success_list_path: str,
    task_name: str,
    task_map: dict,
) -> int:
    """
    롤아웃 결과에서 성공한 에피소드만 추려서 success_list.jsonl 갱신.
    check_success_condition / update_success_list 호출.
    """
    logger.info(f"[Step 2] 성공 필터링 시작")

    # 코드 임포트 (4주차 RFT 관련 파일)
    from b1k.training.rft_utils import (
        check_success_condition,
        create_episode_id,
        update_success_list,
    )

    rollout_path = Path(rollout_output_dir)
    success_count = 0
    total_count = 0

    # 이번 라운드 데이터로 초기화 (누적하려면 "w" → "a")
    open(success_list_path, "w").close()

    for run_dir in sorted(rollout_path.iterdir()):
        if not run_dir.is_dir():
            continue

        total_count += 1

        if check_success_condition(run_dir):
            episode_id = create_episode_id(task_name, task_map, success_count)
            rel_path = str(run_dir.relative_to(rollout_path))
            update_success_list(Path(success_list_path), rel_path, episode_id)
            success_count += 1

    logger.info(f"[Step 2] 필터링 완료: {total_count}개 중 {success_count}개 성공")

    if success_count == 0:
        logger.warning("[Step 2] 성공 에피소드 0개! 이번 라운드 재학습 건너뜀")

    return success_count


# Step 3: RFT 재학습
def run_rft_train(rft_round: int, sft_checkpoint: str) -> str:
    """
    success_list.jsonl에 기록된 성공 데이터만으로 짧게 재학습.
    config: pi_behavior_b1k_fast_rft (config.py에 추가한 것)
    data_loader.py가 success_list.jsonl 읽어서 자동으로 필터링함.

    weight_loader의 PLACEHOLDER를 실제 SFT 체크포인트 경로로 교체 후 실행.
    """
    logger.info(f"[Step 3] RFT 재학습 시작: Round {rft_round}")

    exp_name = f"rft_round_{rft_round}"

    # config.py의 PLACEHOLDER를 실제 체크포인트 경로로 교체
    _patch_checkpoint_in_config(sft_checkpoint)

    subprocess.run([
        "uv", "run", "scripts/train.py",
        "pi_behavior_b1k_fast_rft",        # RFT 전용 config 사용
        f"--exp_name={exp_name}",
        "--overwrite",
    ], check=True)

    new_ckpt_dir = f"./outputs/checkpoints/pi_behavior_b1k_fast_rft/{exp_name}"
    logger.info(f"[Step 3] 재학습 완료 → {new_ckpt_dir}")
    return new_ckpt_dir


def _patch_checkpoint_in_config(checkpoint_path: str):
    """
    config.py의 PLACEHOLDER_REPLACED_BY_TRAIN_RFT를
    실제 체크포인트 경로로 교체.
    """
    config_path = Path("src/b1k/training/config.py")
    content = config_path.read_text()
    patched = re.sub(
        r'"PLACEHOLDER_REPLACED_BY_TRAIN_RFT"',
        f'"{checkpoint_path}"',
        content,
    )
    config_path.write_text(patched)
    logger.info(f"config.py 체크포인트 경로 교체 완료: {checkpoint_path}")


def get_latest_checkpoint(base_dir: str) -> str:
    """가장 최근 저장된 체크포인트 경로 반환"""
    candidates = sorted(Path(base_dir).glob("*/"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"체크포인트 없음: {base_dir}")
    latest = str(candidates[-1])
    logger.info(f"최신 체크포인트: {latest}")
    return latest

# 메인: RFT 반복 루프
def main():
    parser = argparse.ArgumentParser(description="RFT 반복 학습 루프")
    parser.add_argument("--task_name",         type=str, required=True,
                        help="학습할 태스크 이름 (예: turning_on_radio)")
    parser.add_argument("--sft_checkpoint",    type=str, required=True,
                        help="SFT 체크포인트 경로")
    parser.add_argument("--rft_rounds",        type=int, default=3,
                        help="RFT 반복 횟수 (default: 3)")
    parser.add_argument("--rollout_instances", type=int, default=10,
                        help="롤아웃 인스턴스 수 (default: 10)")
    parser.add_argument("--success_list_path", type=str, default="success_list.jsonl",
                        help="성공 에피소드 목록 파일 경로")
    args = parser.parse_args()

    # 태스크별 에피소드 ID 오프셋 (2등팀 방식, 필요하면 추가)
    task_map = {
        "turning_on_radio":   1000,
        "picking_up_trash":   2000,
        "sorting_vegetables": 3000,
        # 나머지 태스크 필요하면 추가
    }

    current_checkpoint = args.sft_checkpoint
    logger.info(f"RFT 시작 | task={args.task_name} | rounds={args.rft_rounds} | SFT={current_checkpoint}")


    # RFT 반복 루프 (기존 SFT 1회 실행을 이 루프로 교체)
    for rft_round in range(1, args.rft_rounds + 1):
        logger.info(f"\n{'='*55}")
        logger.info(f"  RFT Round {rft_round} / {args.rft_rounds}")
        logger.info(f"{'='*55}")

        rollout_dir = f"./outputs/rft/round_{rft_round}/{args.task_name}"

        # Step 1: 롤아웃
        run_rollout(
            task_name=args.task_name,
            rollout_output_dir=rollout_dir,
            num_instances=args.rollout_instances,
        )

        # Step 2: 성공 필터링 → success_list.jsonl 갱신
        success_count = filter_success_episodes(
            rollout_output_dir=rollout_dir,
            success_list_path=args.success_list_path,
            task_name=args.task_name,
            task_map=task_map,
        )

        # 성공 데이터 없으면 건너뜀
        if success_count == 0:
            logger.warning(f"Round {rft_round}: 성공 데이터 없음, 건너뜀")
            continue

        # Step 3: 성공 데이터로 재학습
        new_ckpt_dir = run_rft_train(
            rft_round=rft_round,
            sft_checkpoint=current_checkpoint,
        )

        # 다음 라운드는 이번 결과에서 시작
        current_checkpoint = get_latest_checkpoint(new_ckpt_dir)
        logger.info(f"Round {rft_round} 완료 → 다음 체크포인트: {current_checkpoint}")

    logger.info(f"\n{'='*55}")
    logger.info(f"  RFT 전체 완료! 최종 체크포인트: {current_checkpoint}")
    logger.info(f"{'='*55}")


if __name__ == "__main__":
    main()
