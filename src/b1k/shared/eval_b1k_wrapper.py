"""B1K System 1 Policy Wrapper: Semantic Filtering + Retry Heuristic"""

import logging
import numpy as np
import torch
import dataclasses
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad

logger = logging.getLogger(__name__)
RESIZE_SIZE = 224

# ==========================================
# 1. Semantic 필터 헬퍼 함수 
# ==========================================
def apply_semantic_filter(rgb: np.ndarray, seg_instance: np.ndarray) -> np.ndarray:
    """배경을 지우고 객체 픽셀만 남기는 Semantic 필터"""
    if seg_instance is None:
        return rgb
    seg_instance = np.asarray(seg_instance)
    if seg_instance.ndim == 3:
        seg_instance = seg_instance[..., 0]
    mask = seg_instance != 0
    filtered = np.zeros_like(rgb)
    filtered[mask] = rgb[mask]
    return filtered

# ==========================================
# 2. 필수 Config (System 2 관련 변수 완전 제거)
# ==========================================
@dataclasses.dataclass
class B1KWrapperConfig:
    """Semantic + 리트라이 휴리스틱을 위한 최소 설정"""
    execute_in_n_steps: int = 1
    num_steps: int = 1
    # [Retry 추가] 실패 시 얼마나 뒤로 뺄지/임계값 설정
    recovery_steps: int = 10 
    grasp_threshold: float = 0.01  # 이보다 작게 닫히면 실패로 간주

# ==========================================
# 3. 경량화 Wrapper
# ==========================================
class B1KPolicyWrapper():
    """System 2를 배제하고, Semantic Filter와 Retry 로직만 이식한 순수 래퍼"""

    def __init__(self, policy: BasePolicy, task_id: int | None = None, config: B1KWrapperConfig = None, **kwargs) -> None:
        self.policy = policy
        self.task_id = task_id
        self.config = config if config is not None else B1KWrapperConfig()
        
        # 상태 변수 초기화
        self.current_stage = 0
        self.is_recovering = False
        self.recovery_step_count = 0

    def reset(self):
        """환경 초기화 시 리트라이 상태도 리셋"""
        self.policy.reset()
        self.current_stage = 0
        self.is_recovering = False
        self.recovery_step_count = 0
        logger.info("Policy Reset: Semantic Filtering + Retry Heuristic Mode.")

    def process_obs(self, obs: dict) -> dict:
        """System 1 형태에 Semantic 필터만 추가된 전처리"""
        prop_state = obs["robot_r1::proprio"]
        
        # 1. 원본 RGB
        head_rgb = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_rgb = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_rgb = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]

        # 2. Instance segmentation 데이터
        head_seg = obs.get("robot_r1::robot_r1:zed_link:Camera:0::seg_instance_id", None)
        left_seg = obs.get("robot_r1::robot_r1:left_realsense_link:Camera:0::seg_instance_id", None)
        right_seg = obs.get("robot_r1::robot_r1:right_realsense_link:Camera:0::seg_instance_id", None)

        # 3. Semantic 필터 적용
        head_semantic = apply_semantic_filter(head_rgb, head_seg)
        left_semantic = apply_semantic_filter(left_rgb, left_seg)
        right_semantic = apply_semantic_filter(right_rgb, right_seg)

        # 4. 리사이징 후 반환 (b1k_policy.py에서 받을 때 '_semantic' 이름에 맞춤)
        return {
            "observation/egocentric_camera_semantic": resize_with_pad(head_semantic, RESIZE_SIZE, RESIZE_SIZE),
            "observation/wrist_image_left_semantic": resize_with_pad(left_semantic, RESIZE_SIZE, RESIZE_SIZE),
            "observation/wrist_image_right_semantic": resize_with_pad(right_semantic, RESIZE_SIZE, RESIZE_SIZE),
            "observation/state": prop_state,
            "tokenized_prompt": np.array([self.task_id if self.task_id is not None else -1, self.current_stage], dtype=np.int32),
            "tokenized_prompt_mask": np.array([True, True], dtype=bool),
        }

    def act(self, obs: dict) -> torch.Tensor:
        """[핵심] System 2 배제, 리트라이 휴리스틱 최우선 적용 구역"""
        
        # 1. 실패 감지 (Retry Heuristic 발동 조건)
        gripper_width = obs["robot_r1::proprio"][22]
        
        if not self.is_recovering and gripper_width < self.config.grasp_threshold:
            self.is_recovering = True
            self.recovery_step_count = self.config.recovery_steps
            logger.info("⚠️ 잡기 실패 감지! 리트라이(후진) 동작을 수행합니다.")

        # 2. 리트라이 동작 수행 (if: 복구 모드인 경우 모델 추론 무시)
        if self.is_recovering and self.recovery_step_count > 0:
            retry_action = np.zeros(23)
            retry_action[0] = -0.15  # x축 강제 후진 (팔 뒤로 빼기)
            retry_action[22] = 1.0   # 그리퍼 강제 열기
            
            self.recovery_step_count -= 1
            if self.recovery_step_count == 0:
                self.is_recovering = False
                logger.info("✅ 복구 동작 완료. 다시 모델 제어로 복귀합니다.")
                
            return torch.from_numpy(retry_action).float()

        # 3. 정상 동작 (else: 순수 System 1 추론)
        model_input = self.process_obs(obs)
        output = self.policy.infer(model_input)
        
        # 스테이지 업데이트 (즉시 반영)
        if "subtask_logits" in output:
            self.current_stage = int(np.argmax(output["subtask_logits"]))

        # 액션 반환
        actions = output["actions"]
        current_action = actions[0] if len(actions.shape) == 3 else actions
        if isinstance(current_action, np.ndarray) and len(current_action.shape) > 1:
            current_action = current_action[0]

        return torch.from_numpy(current_action[:23]).float()
