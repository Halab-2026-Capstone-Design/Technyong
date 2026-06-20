# Evaluation Report: Task 0 (turning_on_radio) — Trained Checkpoint
**작성일**: 2026-05-10  
**평가 시간**: 10:22 ~ 11:49 KST (약 1시간 27분)

---

## 1. 평가에 사용한 파일

### Policy Server
| 항목 | 경로 |
|------|------|
| 서버 스크립트 | `Technyong/scripts/serve_b1k.py` |
| Wrapper 구현 | `Technyong/src/b1k/shared/eval_b1k_wrapper.py` |
| Checkpoint Switcher | `behavior-1k-solution/src/b1k/policies/checkpoint_switcher.py` |

### Checkpoint
| 항목 | 값 |
|------|-----|
| Config 이름 | `pi_behavior_b1k_task0000_train` |
| 체크포인트 경로 | `Technyong/outputs/checkpoints/pi_behavior_b1k_task0000_train/task000_turning_on_radio_sft_20260509_2037/4999` |
| 훈련 스텝 | 4999 / 5000 (거의 완료) |
| Base weight | `models/behavior_submission/checkpoint_1/params` (submission checkpoint 1 기반 fine-tune) |

### Eval 스크립트 & 환경
| 항목 | 경로/값 |
|------|---------|
| Eval 실행 스크립트 | `/tmp/run_eval_task0_trained.sh` |
| Eval 코어 스크립트 | `behavior-1k-solution/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py` |
| Task instances 정의 | `BEHAVIOR-1K/datasets/2025-challenge-task-instances/metadata/test_instances.csv` |
| Python 환경 | `/root/miniconda3/envs/behavior` (JAX 0.6.2) |
| 결과 로그 | `/tmp/eval_task0_trained.log` |
| Metrics 저장 | `behavior-1k-solution/BEHAVIOR-1K/eval_logs/task0_trained/metrics/` |
| 영상 저장 | `behavior-1k-solution/BEHAVIOR-1K/eval_logs/task0_trained/videos/` |

---

## 2. 평가 조건

### Task 설정
| 항목 | 값 |
|------|-----|
| Task 이름 | `turning_on_radio` |
| Task ID | 0 |
| 평가 Instances | 인덱스 0~9 (전역 ID: 242, 295, 211, 203, 109, 181, 197, 187, 214, 139) |
| 최대 스텝 수 | 4,300 steps / instance (143.3초) |
| 총 시간 예산 | 약 286초 (8,600 steps 기준, normalized_time 최대 1.0) |

### Policy Server 설정
| 항목 | 값 |
|------|-----|
| Port | 8000 |
| Protocol | WebSocket |
| `control_mode` | `receding_horizon` |
| `actions_to_execute` | 26 |
| `actions_to_keep` | 4 |
| `execute_in_n_steps` | 20 |
| `num_steps` | 20 |
| `action_horizon` | 5 |
| `temporal_ensemble_max` | 3 |
| `history_len` | 3 |
| `votes_to_promote` | 2 |
| Correction Rules | **미적용** (Technyong 버전 wrapper는 correction/eval tricks 제외) |
| Task-checkpoint mapping | 없음 (단일 체크포인트 모드) |

### 환경 변수
```
OMNIGIBSON_HEADLESS=1
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
OMNIGIBSON_DATA_PATH=/home/data/Technyong_workspace/BEHAVIOR-1K/datasets
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
MPLCONFIGDIR=/tmp/matplotlib-behavior
```

### 훈련 설정 (참고)
| 항목 | 값 |
|------|-----|
| 훈련 데이터 | `behavior-1k-solution/data/behavior_224_rgb` |
| Task 0 훈련 에피소드 수 | **200개** |
| Freeze vision backbone | True (action expert만 학습) |
| Learning rate | Cosine decay: 2e-5 → 1e-6 (warmup 100, decay 3,000 steps) |
| Batch size | 16 |
| 총 훈련 스텝 | 5,000 |

---

## 3. 평가 결과 분석

### 전체 요약
| 항목 | 결과 |
|------|------|
| 총 Instances | 10 |
| 성공 | **0** |
| 실패 (timeout) | **10** |
| 성공률 | **0.0%** |
| 평균 qscore | **0.0** |

### Instance별 세부 결과

| Instance | q_score | norm_dist_base | norm_dist_left | norm_dist_right | norm_time | 비고 |
|----------|---------|----------------|----------------|-----------------|-----------|------|
| 242 | 0.000 | 0.239 | 0.315 | 0.383 | 0.500 | 이동 거리 매우 적음 |
| 295 | 0.000 | 1.996 | 2.582 | 3.085 | 0.500 | 이동 거리 많음 |
| 211 | 0.000 | 0.368 | 0.441 | 0.596 | 0.500 | |
| 203 | 0.000 | 0.334 | 0.344 | 0.458 | 0.500 | 이동 거리 매우 적음 |
| 109 | 0.000 | 0.510 | 0.677 | 0.990 | 0.500 | |
| 181 | 0.000 | 0.345 | 0.581 | 0.622 | 0.500 | 이동 거리 매우 적음 |
| 197 | 0.000 | 1.035 | 1.601 | 1.676 | 0.500 | |
| 187 | 0.000 | 0.967 | 1.238 | 1.484 | 0.500 | |
| 214 | 0.000 | 0.390 | 0.612 | 0.784 | 0.500 | |
| 139 | 0.000 | 2.654 | 3.043 | 3.626 | 0.500 | 이동 거리 가장 많음 |

> **norm_dist**: 로봇이 이동한 총 거리를 정규화한 값. 값이 클수록 많이 움직임.  
> **norm_time**: 사용된 시간 / 최대 시간 예산. 전 instance 0.500 → 모두 타임아웃.

### 패턴 분석
- **norm_time = 0.500 (전 instance 동일)**: 모든 instance가 4,300 스텝을 전부 소진하고 타임아웃으로 종료. 조기 성공 없음.
- **이동 거리 패턴**: 6개 instance(242, 203, 181, 211, 214, 109)에서 `norm_dist_base < 1.0` → 로봇이 라디오 근처까지 접근조차 못하거나 제자리 반복
- **이동 거리 많은 instance**(295, 139, 197, 187): 로봇은 움직였으나 태스크 완료 실패 → 올바른 action sequence를 만들어내지 못함

---

## 4. 원인 분석

### 원인 1: Correction Rules 미적용 (가장 유력)

Technyong `serve_b1k.py`가 사용하는 `Technyong/src/b1k/shared/eval_b1k_wrapper.py`는 submission 버전과 구조가 다릅니다.

```
# Technyong eval_b1k_wrapper.py 265번 줄 주석
"Original 1st-place execution shell, minus correction/eval tricks."
```

즉, **의도적으로 correction rules와 eval tricks를 제거**한 버전입니다.

반면 `behavior-1k-solution/src/b1k/shared/eval_b1k_wrapper.py`에는 다음이 포함됩니다:
- `apply_eval_tricks: bool = True` → 기본값으로 correction rules 활성화
- `MIN_STAGE_FOR_CLOSURE`: task 0에서 **stage 2 이전에는 그리퍼 닫기 차단** (`left: 2, right: 2`)
- `task0_stage4_reset_to_stage2`: stage 4 도달 시 stage 2로 리셋 (라디오 태스크 특화)

Correction rules 없이는 task 0에서 그리퍼가 잘못된 타이밍에 닫혀 태스크 진행이 막힐 수 있습니다.

### 원인 2: Receding Horizon 실행 모드의 부적합성

현재 사용 중인 `control_mode=receding_horizon` 모드:
- 액션 큐가 소진되면 새로 inference → 이전 액션과의 연속성 없음
- Rolling inpainting 미사용 → 이전 컨텍스트 활용 안 됨

Submission에서 1위를 달성한 `rolling_inpainting` 모드와 달리, receding horizon은 각 예측이 독립적으로 이루어집니다. Task 0처럼 연속적인 조작이 필요한 태스크에서 불리합니다.

### 원인 3: Fine-tune 데이터 도메인 불일치

```
훈련 데이터: behavior_224_rgb, task 0 에피소드 200개
훈련 데이터 경로: behavior-1k-solution/data/behavior_224_rgb
```

훈련 데이터가 **eval instance들(242, 295, 211...)**과 동일한 씬에서 수집되었는지 확인이 필요합니다. 씬 구성(라디오 위치, 주변 오브젝트 배치)이 훈련 데이터와 다르면 일반화 실패가 발생합니다. 특히 `norm_dist_base < 0.4`인 instance들(242, 203, 181)은 로봇이 아예 접근하지 못한 것으로 보입니다.

### 원인 4: Vision Backbone Freeze의 한계

```python
freeze_vision_backbone=True  # 비전은 고정, action expert만 학습
```

Submission checkpoint 1은 50개 전체 task를 학습한 general model입니다. Task 0만 200 에피소드로 fine-tune할 때 vision backbone을 고정하면:
- 시각적 특징 추출은 general model 그대로 → task 0 특화 장면 인식 개선 없음
- action expert만 조정되므로 개선 폭이 제한됨

### 원인 5: Submission Checkpoint 1 기반의 한계

Task 0은 `task_checkpoint_mapping.json` 기준으로 **checkpoint_2**에 할당되어 있습니다.

```json
"checkpoint_2": { "tasks": [1,45,0,16,12,9,18,20,21,22,30,7,8,17,26,43] }
```

Fine-tune의 base weight로 checkpoint_1을 사용했는데, checkpoint_2가 task 0에 더 적합할 수 있습니다.

---

## 5. 개선 방안

### 방안 A: Correction Rules 활성화 (즉시 적용 가능)

`behavior-1k-solution/scripts/serve_b1k.py` + `behavior-1k-solution` 버전 wrapper를 사용하여 `apply_eval_tricks=True`로 평가 재실행.

```bash
# behavior-1k-solution 버전 서버 (correction rules 포함)
python serve_b1k.py \
  --apply-eval-tricks True \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_task0000_train \
  --policy.dir .../4999
```

### 방안 B: Rolling Inpainting 모드로 전환

Technyong wrapper에서 `control_mode=rolling_inpainting`으로 변경:
- 이전 액션을 다음 inference의 초기값으로 활용 → 연속성 향상
- Submission 1위 방식과 동일한 실행 패턴

### 방안 C: Base Checkpoint 변경

Fine-tune 시작점을 checkpoint_1 대신 **checkpoint_2**로 변경:
```python
weight_loader=weight_loaders.PiBehaviorWeightLoader(
    "/home/data/.../checkpoint_2/params"  # task 0 담당 checkpoint
)
```

### 방안 D: Submission Checkpoint 동일 조건 비교 평가

현재 trained checkpoint 성능의 기준선 확인을 위해 **submission checkpoint_2**로 동일한 eval 조건(correction rules 없음, receding_horizon)에서 평가 후 비교.

### 방안 E: 훈련 개선
- `freeze_vision_backbone=False`로 전체 fine-tune 시도 (더 많은 데이터 필요)
- 훈련 데이터 확대: 200 → 500+ 에피소드
- eval instance scene들(242, 295, ...)이 훈련 데이터에 포함되어 있는지 확인

---

## 6. 결론

**핵심 결론: trained checkpoint (step 4999) 단독으로는 qscore 0.0.**

가장 즉각적인 원인은 **Correction Rules 미적용**입니다. Technyong 버전 wrapper는 명시적으로 correction/eval tricks를 제거한 버전을 사용하고 있어, task 0에 필요한 그리퍼 타이밍 제어(MIN_STAGE_FOR_CLOSURE)와 stage 리셋 로직(task0_stage4_reset_to_stage2)이 동작하지 않습니다.

두 번째 원인은 **receding_horizon 모드**입니다. Submission에서 검증된 rolling_inpainting 방식 대비 각 예측이 독립적이어서 연속 조작 태스크에 불리합니다.

Fine-tune 자체의 효과를 검증하려면, **동일한 서버 설정(correction rules 포함, rolling_inpainting)**으로 submission checkpoint_2와 trained checkpoint를 A/B 비교해야 합니다.

### 권장 다음 단계
1. `behavior-1k-solution` 버전 wrapper + `apply_eval_tricks=True` + `rolling_inpainting`으로 trained checkpoint 재평가
2. 동일 조건에서 submission checkpoint_2 평가 → baseline 확립
3. 두 결과 비교로 fine-tune의 실제 효과 측정
4. 성능 차이 확인 후 훈련 데이터 / base checkpoint / fine-tune 전략 조정
