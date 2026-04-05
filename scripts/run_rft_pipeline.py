import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def run_rollout_with_server(task_name: str, rollout_output_dir: str, num_instances: int, current_ckpt: str):
    """
    [핵심] 서버(serve_b1k)를 백그라운드에서 켜고, 롤아웃이 끝나면 안전하게 끕니다.
    매 라운드마다 똑똑해진 최신 체크포인트(current_ckpt)로 뇌를 교체하여 서버를 엽니다.
    """
    logger.info(f"🧠 AI 서버 시작 중... (체크포인트: {current_ckpt})")
    
    # 1. 서버 백그라운드 실행
    server_process = subprocess.Popen([
        "python", "scripts/serve_b1k.py",
        "--save_rollout=True",
        f"--policy.dir={current_ckpt}",
        "--policy.config=pi_behavior_b1k_fast_rft"
    ])
    
    # 서버가 웹소켓을 완전히 열 때까지 20초 대기
    time.sleep(20) 

    logger.info(f"▶️ [Step 1] 롤아웃 시작: task={task_name}, instances={num_instances}")
    Path(rollout_output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 2. 옴니깁슨 롤아웃 실행 (켜져있는 서버로 데이터 전송)
        subprocess.run([
            "python", "scripts/eval_custom.py",
            "policy=websocket",
            "save_rollout=true",
            "perturb_pose=true",
            f"task.name={task_name}",
            f"log_path={rollout_output_dir}",
            "use_parallel_evaluator=false",
            "parallel_evaluator_start_idx=0",
            f"parallel_evaluator_end_idx={num_instances}",
            "model.port=8000",
            "env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper",
        ], check=True)
    finally:
        # 3. 롤아웃 종료 후 서버 강제 종료 (다음 라운드를 위해 포트 비우기)
        logger.info("🛑 롤아웃 완료. AI 서버를 종료합니다.")
        server_process.terminate()
        server_process.wait()

def filter_success_episodes(rollout_output_dir: str, success_list_path: str, task_name: str, task_map: dict) -> int:
    """롤아웃 결과에서 성공한 에피소드만 추려서 success_list.jsonl 갱신"""
    from b1k.training.rft_utils import check_success_condition, create_episode_id, update_success_list
    rollout_path = Path(rollout_output_dir)
    success_count, total_count = 0, 0
    
    # 이번 라운드 데이터로 초기화
    #open(success_list_path, "w").close()

    for run_dir in sorted(rollout_path.iterdir()):
        if not run_dir.is_dir(): continue
        total_count += 1
        if check_success_condition(run_dir):
            episode_id = create_episode_id(task_name, task_map, success_count)
            update_success_list(Path(success_list_path), str(run_dir.relative_to(rollout_path)), episode_id)
            success_count += 1
            
    logger.info(f"[Step 2] 필터링 완료: {total_count}개 중 {success_count}개 성공")
    return success_count

def run_rft_train(rft_round: int, sft_checkpoint: str, task_name: str) -> str:
    """
    [핵심] 환경 변수를 통해 안전하게 체크포인트 경로를 config.py에 주입하고 1등팀의 train.py를 호출합니다.
    """
    logger.info(f"[Step 3] RFT 재학습 시작: Round {rft_round}")
    exp_name = f"rft_round_{rft_round}_{task_name}"

    # 환경 변수에 체크포인트 경로 심기
    env = os.environ.copy()
    env["RFT_CKPT_PATH"] = sft_checkpoint

    # 1등 팀의 원본 train.py를 우리의 RFT 설정으로 실행
    subprocess.run([
        "uv", "run", "scripts/train.py",
        "pi_behavior_b1k_fast_rft",
        f"--exp_name={exp_name}",
        "--overwrite",
    ], env=env, check=True)

    return f"./outputs/checkpoints/pi_behavior_b1k_fast_rft/{exp_name}"

def get_latest_checkpoint(base_dir: str) -> str:
    """가장 최근 저장된 체크포인트(방금 학습한 결과) 경로 반환"""
    candidates = sorted(Path(base_dir).glob("*/"), key=os.path.getmtime)
    if not candidates: raise FileNotFoundError(f"체크포인트 없음: {base_dir}")
    return str(candidates[-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--rft_rounds", type=int, default=3)
    parser.add_argument("--rollout_instances", type=int, default=10)
    parser.add_argument("--success_list_path", type=str, default="success_list.jsonl")
    args = parser.parse_args()

    task_map = {"picking_up_trash" : 1000,          # 쓰레기 줍기
                "collecting_aluminum_cans" : 2000,  # 알루미늄 캔 수거하기
                "storing_groceries" : 3000,            # 식료품 보관하기
                "cleaning_shoes" : 4000,               # 신발 닦기
                "putting_away_toys" : 5000,       # 장난감 정리하기
                "throwing_away_leftovers" : 6000,    # 남은 음식 버리기
                "organizing_cleaning_supplies" : 7000, # 청소 용품 정리하기
                "setting_the_table" : 8000,           # 식탁 차리기
                "packing_a_box" : 9000,            # 상자 포장하기
                "washing_dishes" : 10000,                  # 설거지하기
                "wiping_the_table" : 11000,                # 식탁 닦기
                "vacuuming_the_floor" : 12000,             # 진공청소기 돌리기
                "sweeping_the_floor" : 13000,              # 바닥 쓸기
                "mopping_the_floor" : 14000,               # 바닥 걸레질하기
                "dusting_shelves" : 15000,                 # 선반 먼지 털기
                "cleaning_windows" : 16000,                # 창문 닦기
                "making_the_bed" : 17000,                  # 침대 정리하기
                "watering_plants" : 18000,                 # 화분에 물 주기
                "setting_up_a_tent" : 19000,               # 텐트 설치하기
                "taking_out_the_trash" : 20000,            # 쓰레기 내다 버리기
                "sorting_mail" : 21000,                    # 우편물 분류하기 
                "turning_on_radio" : 22000,                # 라디오 켜기
                "opening_the_fridge" : 23000,              # 냉장고 열기 (정리)
                "washing_clothes" : 24000,                 # 세탁기 돌리기
                "drying_clothes" : 25000,                  # 건조기 돌리기
                "folding_laundry" : 26000,                 # 빨래 개기
                "ironing_clothes" : 27000,                # 옷 다림질하기
                "organizing_a_closet" : 28000,             # 옷장 정리하기
                "packing_a_suitcase" : 29000,              # 여행 가방 싸기
                "unpacking_a_suitcase" : 30000,            # 여행 가방 풀기
                "arranging_books" : 31000,                 # 책장에 책 꽂기
                "cleaning_the_bathroom" :329000,           # 화장실 청소하기
                "cleaning_the_kitchen" : 33000,            # 주방 청소하기
                "cleaning_the_microwave" : 34000,          # 전자레인지 청소하기
                "cleaning_the_oven" : 35000,               # 오븐 청소하기
                "loading_the_dishwasher" : 36000,          # 식기세척기에 그릇 넣기
                "unloading_the_dishwasher" : 37000,        # 식기세척기에서 그릇 빼기
                "stocking_the_pantry" : 38000,             # 식료품 창고 채우기
                "boiling_water" : 39000,                   # 물 끓이기
                "preparing_a_salad" : 40000,               # 샐러드 준비하기
                "making_a_sandwich" : 41000,               # 샌드위치 만들기
                "chopping_an_onion" : 42000,               # 양파 썰기
                "setting_up_a_coffee_station" : 43000,     # 커피 스테이션 세팅하기
                "feeding_a_pet" : 44000,                   # 반려동물 먹이 주기
                "cleaning_a_litter_box" : 45000,           # 배변 훈련함(고양이 화장실) 치우기
                "replacing_a_light_bulb" : 46000,          # 전구 교체하기
                "assembling_furniture" : 47000,            # 가구 조립하기
                "mowing_the_lawn" : 48000,                 # 잔디 깎기
                "washing_a_car" : 49000,                   # 세차하기
                "cleaning_a_bicycle" : 50000,              # 자전거 닦기
               }

    current_checkpoint = args.sft_checkpoint

    for rft_round in range(1, args.rft_rounds + 1):
        logger.info(f"\n{'='*55}\n  RFT Round {rft_round} / {args.rft_rounds}\n{'='*55}")
        rollout_dir = f"./outputs/rft/round_{rft_round}/{args.task_name}"

        # 1. 서버 켜기 -> 롤아웃 -> 서버 끄기
        run_rollout_with_server(args.task_name, rollout_dir, args.rollout_instances, current_checkpoint)

        # 2. 성공 데이터 필터링
        success_count = filter_success_episodes(rollout_dir, args.success_list_path, args.task_name, task_map)
        if success_count == 0:
            logger.warning(f"Round {rft_round}: 성공 데이터 없음, 건너뜀")
            continue

        # 3. 성공 데이터로 재학습 진행 (새로운 체크포인트 생성)
        new_ckpt_dir = run_rft_train(rft_round, current_checkpoint, args.task_name)
        
        # 4. 다음 라운드를 위해 뇌(체크포인트) 업데이트
        current_checkpoint = get_latest_checkpoint(new_ckpt_dir)
        logger.info(f"Round {rft_round} 완료 → 다음 체크포인트: {current_checkpoint}")

if __name__ == "__main__":
    main()
