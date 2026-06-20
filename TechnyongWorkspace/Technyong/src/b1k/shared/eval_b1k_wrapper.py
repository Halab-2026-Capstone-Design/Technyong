"""B1K policy wrapper with rolling inpainting and Comet-style control modes."""

import logging
import numpy as np
import torch
import dataclasses
from collections import deque

from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from b1k.policies.b1k_policy import extract_state_from_proprio
from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.shared.correction_rules import apply_correction_rules, check_gripper_variation

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224
LEFT_GRIPPER_IDX = 14
RIGHT_GRIPPER_IDX = 22
SUPPORTED_CONTROL_MODES = {
    "rolling_inpainting",
    "receding_horizon",
    "temporal_ensemble",
    "receding_temporal",
    "hybrid_rolling_temporal",
}


@dataclasses.dataclass
class B1KWrapperConfig:
    """Configuration for B1K policy wrapper execution parameters."""
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True
    control_mode: str = "rolling_inpainting"
    action_horizon: int = 5
    temporal_ensemble_max: int = 3


class B1KPolicyWrapper():
    """B1K policy wrapper for PI_BEHAVIOR models with action compression, rolling inpainting, and stage voting."""
    
    def __init__(
        self, 
        policy: BasePolicy,
        text_prompt: str = "PI_BEHAVIOR model (task-conditioned)",  # Not used, kept for compatibility
        action_horizon: int = 30,
        task_id: int | None = None,
        config: B1KWrapperConfig = None,
        checkpoint_switcher = None,
    ) -> None:
        self.base_policy = policy
        self.policy = policy
        self.checkpoint_switcher = checkpoint_switcher
        self.text_prompt = text_prompt
        self.action_horizon = action_horizon
        self.config = config if config is not None else B1KWrapperConfig()
        if self.config.control_mode not in SUPPORTED_CONTROL_MODES:
            raise ValueError(
                f"Unsupported control_mode={self.config.control_mode!r}. "
                f"Expected one of {sorted(SUPPORTED_CONTROL_MODES)}"
            )
        
        # Validate configuration
        if self.config.actions_to_execute + self.config.actions_to_keep > self.action_horizon:
            raise ValueError(
                f"actions_to_execute + actions_to_keep exceeds action_horizon"
            )
        
        # PI_BEHAVIOR specific (always True for B1K)
        self.task_id = task_id
        self.current_stage = 0
        self.prediction_history = deque([], maxlen=self.config.history_len)
        
        # Control loop variables
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
        self.action_queue = deque()
        self.temporal_action_queue = deque(maxlen=self.config.temporal_ensemble_max)
    
    def reset(self):
        """Reset policy state."""
        self.policy.reset()
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
        self.action_queue.clear()
        self.temporal_action_queue.clear()
        self.current_stage = 0
        self.prediction_history.clear()
        logger.info(f"Policy reset - Task ID: {self.task_id}, Action horizon: {self.action_horizon}")
    
    def _handle_task_change(self, new_task_id):
        """Handle task ID change by switching checkpoint and resetting state."""
        if self.task_id != new_task_id:
            old_task_id = self.task_id
            self.task_id = new_task_id
            
            logger.info(f"🔄 Task change detected: {old_task_id} → {new_task_id} (max stages: {TASK_NUM_STAGES[new_task_id]})")
            
            if self.checkpoint_switcher:
                new_policy = self.checkpoint_switcher.get_policy_for_task(new_task_id)
                if new_policy is not self.policy:
                    logger.info(f"📦 Switching checkpoint: task {old_task_id} → {new_task_id}")
                    self.base_policy = new_policy
                    self.policy = new_policy
                    self.policy.reset()
            
            self.current_stage = 0
            self.prediction_history.clear()
            self.last_actions = None
            self.action_index = 0
            self.next_initial_actions = None
            self.action_queue.clear()
            self.temporal_action_queue.clear()

    def process_obs(self, obs: dict) -> dict:
        """Process observation to match model input format."""
        prop_state = obs["robot_r1::proprio"]
        
        head_original = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_original = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_original = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
        
        # Resize images
        head_resized = resize_with_pad(head_original, RESIZE_SIZE, RESIZE_SIZE)
        left_resized = resize_with_pad(left_original, RESIZE_SIZE, RESIZE_SIZE)
        right_resized = resize_with_pad(right_original, RESIZE_SIZE, RESIZE_SIZE)
        
        return {
            "observation/egocentric_camera_semantic": head_resized,
            "observation/wrist_image_left_semantic": left_resized,
            "observation/wrist_image_right_semantic": right_resized,
            "observation/state": prop_state,
            "prompt": self.text_prompt,
        }
    
    def update_current_stage(self, predicted_subtask_logits):
        """Update current stage using majority voting."""
        if self.task_id is None:
            return
            
        max_stage = TASK_NUM_STAGES[self.task_id] - 1
        predicted_stage = int(np.argmax(predicted_subtask_logits))
        
        if predicted_stage > max_stage:
            predicted_stage = max_stage
        
        self.prediction_history.append(predicted_stage)
        
        if len(self.prediction_history) == self.config.history_len:
            next_stage = self.current_stage + 1
            
            if next_stage <= max_stage:
                votes_for_next = sum(1 for pred in self.prediction_history if pred == next_stage)
                votes_to_skip = sum(1 for pred in self.prediction_history if pred == next_stage + 1)
                votes_to_go_back = sum(1 for pred in self.prediction_history if pred == self.current_stage - 1)
                
                if votes_for_next >= self.config.votes_to_promote:
                    old_stage = self.current_stage
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                    logger.info(f"⬆️  Stage advanced: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
                elif votes_to_skip == self.config.history_len:
                    old_stage = self.current_stage
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                    logger.info(f"⏭️  Stage skipped: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
                elif votes_to_go_back == self.config.history_len and self.current_stage > 0:
                    old_stage = self.current_stage
                    self.current_stage -= 1
                    self.prediction_history.clear()
                    logger.info(f"⬅️  Stage went back: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
    
    def prepare_batch_for_pi_behavior(self, batch):
        """Prepare batch for PI_BEHAVIOR model by adding task_id and current_stage."""
        task_id = self.task_id if self.task_id is not None else -1
        batch_copy = batch.copy()
        if "prompt" in batch_copy:
            del batch_copy["prompt"]
        
        batch_copy["tokenized_prompt"] = np.array([task_id, self.current_stage], dtype=np.int32)
        batch_copy["tokenized_prompt_mask"] = np.array([True, True], dtype=bool)
        batch_copy["subtask_state"] = np.array(self.current_stage, dtype=np.int32)
        
        return batch_copy
    
    def _interpolate_actions(self, actions, target_steps):
        """Interpolate actions using cubic spline."""
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, len(actions)-1, len(actions))
        target_indices = np.linspace(0, len(actions)-1, target_steps)
        
        interpolated = np.zeros((target_steps, actions.shape[1]))
        for dim in range(actions.shape[1]):
            f = interp1d(original_indices, actions[:, dim], kind='cubic')
            interpolated[:, dim] = f(target_indices)
        
        return interpolated

    def _apply_eval_tricks(self, obs: dict, actions: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply first-place correction rules before any control-mode scheduling."""
        should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute
        if not self.config.apply_eval_tricks:
            return actions, should_compress

        if self.task_id is not None:
            current_state = extract_state_from_proprio(obs["robot_r1::proprio"])
            actions_before = actions.copy()
            actions, corrected_stage = apply_correction_rules(
                self.task_id, self.current_stage, current_state, actions
            )

            if corrected_stage != self.current_stage:
                logger.info(
                    f"🔧 Correction rule: Stage corrected {self.current_stage} → "
                    f"{corrected_stage} (task {self.task_id}, step {self.step_count})"
                )
                self.current_stage = corrected_stage
                self.prediction_history.clear()

            if not np.allclose(actions_before, actions, rtol=1e-3):
                max_diff = np.max(np.abs(actions_before - actions))
                logger.info(
                    f"🔧 Correction rule: Actions modified (max diff: {max_diff:.4f}, "
                    f"task {self.task_id}, stage {self.current_stage})"
                )

        if should_compress:
            has_high_variation, mean_var, max_var = check_gripper_variation(
                actions, self.config.actions_to_execute
            )
            if has_high_variation:
                should_compress = False
                logger.info(
                    f"🔧 Gripper variation: Compression disabled "
                    f"(mean: {mean_var:.4f}, max: {max_var:.4f})"
                )

        return actions, should_compress

    def _infer_actions(
        self, obs: dict, *, use_inpainting: bool, apply_tricks: bool = True
    ) -> tuple[np.ndarray, dict, bool]:
        """Run the policy and return actions as a 2D numpy array."""
        model_input = self.process_obs(obs)
        model_input = self.prepare_batch_for_pi_behavior(model_input)

        if use_inpainting and self.next_initial_actions is not None:
            model_input["initial_actions"] = self.next_initial_actions

        if "initial_actions" in model_input and model_input["initial_actions"] is not None:
            output = self.policy.infer(model_input, initial_actions=model_input["initial_actions"])
        else:
            output = self.policy.infer(model_input)

        actions = output["actions"]
        if len(actions.shape) == 3:
            actions = actions[0]
        if actions.shape[1] > 23:
            actions = actions[:, :23]

        self.prediction_count += 1
        if apply_tricks:
            actions, should_compress = self._apply_eval_tricks(obs, actions)
        else:
            should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute

        if "subtask_logits" in output:
            self.update_current_stage(output["subtask_logits"])

        return actions, output, should_compress

    def _prepare_rolling_actions(self, actions: np.ndarray, should_compress: bool) -> None:
        """Prepare an action chunk using the original rolling-inpainting path."""
        actions_to_execute = self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
        execute_steps = self.config.execute_in_n_steps

        inpainting_start = actions_to_execute
        inpainting_end = inpainting_start + self.config.actions_to_keep

        if len(actions) >= inpainting_end:
            self.next_initial_actions = actions[inpainting_start:inpainting_end].copy()
        else:
            self.next_initial_actions = None

        self.last_actions = actions[:actions_to_execute].copy()

        if should_compress:
            compressed_actions = self._interpolate_actions(self.last_actions, execute_steps)
            compression_factor = actions_to_execute / execute_steps
            compressed_actions[:, :3] *= compression_factor
            self.last_actions = compressed_actions

        self.action_index = 0

        if self.prediction_count % 10 == 0:
            compression_status = f"compressed {actions_to_execute}→{execute_steps}" if should_compress else f"uncompressed ({execute_steps})"
            logger.info(
                f"🎯 Prediction #{self.prediction_count} | Actions: {compression_status} | "
                f"Inpainting: {self.next_initial_actions is not None}"
            )

    def _next_rolling_inpainting_action(self, obs: dict) -> np.ndarray:
        """Original 1st-place execution shell with optional correction/eval tricks."""
        if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
            actions, _, should_compress = self._infer_actions(obs, use_inpainting=True)
            self._prepare_rolling_actions(actions, should_compress)

        if self.action_index >= len(self.last_actions):
            self.action_index = 0

        current_action = self.last_actions[self.action_index]
        self.action_index += 1
        return current_action

    def _next_receding_horizon_action(self, obs: dict) -> np.ndarray:
        """Execute one queued policy rollout action at a time."""
        if not self.action_queue:
            actions, _, _ = self._infer_actions(obs, use_inpainting=False)
            horizon = min(self.config.actions_to_execute, len(actions))
            self.action_queue = deque(actions[:horizon].copy())
            self.next_initial_actions = None

        return self.action_queue.popleft()

    def _ensemble_current_actions(self, action_sequences: list[deque], latest_action: np.ndarray) -> np.ndarray:
        """Average current actions while preserving the freshest gripper commands."""
        current_actions = np.empty((len(action_sequences), latest_action.shape[0]), dtype=latest_action.dtype)
        for idx, sequence in enumerate(action_sequences):
            current_actions[idx] = sequence.popleft()

        k = 0.005
        exp_weights = np.exp(k * np.arange(current_actions.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()
        final_action = (current_actions * exp_weights[:, None]).sum(axis=0)

        if final_action.shape[0] > RIGHT_GRIPPER_IDX:
            final_action[LEFT_GRIPPER_IDX] = latest_action[LEFT_GRIPPER_IDX]
            final_action[RIGHT_GRIPPER_IDX] = latest_action[RIGHT_GRIPPER_IDX]

        return final_action

    def _next_temporal_ensemble_action(self, obs: dict) -> np.ndarray:
        """Comet-style temporal ensemble: re-predict every step and smooth action streams."""
        actions, _, _ = self._infer_actions(obs, use_inpainting=False)
        self.temporal_action_queue.append(deque(actions.copy()))

        action_sequences = [sequence for sequence in self.temporal_action_queue if len(sequence) > 0]
        latest_action = actions[0]
        final_action = self._ensemble_current_actions(action_sequences, latest_action)
        self.temporal_action_queue = deque(
            [sequence for sequence in action_sequences if len(sequence) > 0],
            maxlen=self.config.temporal_ensemble_max,
        )
        self.next_initial_actions = None
        return final_action

    def _next_receding_temporal_action(self, obs: dict) -> np.ndarray:
        """Comet-style periodic replanning with temporal ensemble across recent rollouts."""
        if self.step_count % self.config.action_horizon == 0 or not self.temporal_action_queue:
            actions, _, _ = self._infer_actions(obs, use_inpainting=False)
            horizon = min(self.config.actions_to_execute, len(actions))
            self.temporal_action_queue.append(deque(actions[:horizon].copy()))
            latest_action = actions[0]
        else:
            latest_sequence = self.temporal_action_queue[-1]
            latest_action = latest_sequence[0] if len(latest_sequence) > 0 else np.zeros(23, dtype=np.float32)

        action_sequences = [sequence for sequence in self.temporal_action_queue if len(sequence) > 0]
        final_action = self._ensemble_current_actions(action_sequences, latest_action)
        self.temporal_action_queue = deque(
            [sequence for sequence in action_sequences if len(sequence) > 0],
            maxlen=self.config.temporal_ensemble_max,
        )
        self.next_initial_actions = None
        return final_action

    def _next_hybrid_rolling_temporal_action(self, obs: dict) -> np.ndarray:
        """Rolling inpainting plus temporal smoothing across recent corrected chunks."""
        if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
            actions, _, should_compress = self._infer_actions(obs, use_inpainting=True)
            self._prepare_rolling_actions(actions, should_compress)
            self.temporal_action_queue.append(deque(self.last_actions.copy()))

        action_sequences = [sequence for sequence in self.temporal_action_queue if len(sequence) > 0]
        if not action_sequences:
            self.action_index = 0
            return self._next_rolling_inpainting_action(obs)

        latest_sequence = self.temporal_action_queue[-1]
        latest_action = latest_sequence[0] if len(latest_sequence) > 0 else self.last_actions[self.action_index]
        final_action = self._ensemble_current_actions(action_sequences, latest_action)
        self.temporal_action_queue = deque(
            [sequence for sequence in action_sequences if len(sequence) > 0],
            maxlen=self.config.temporal_ensemble_max,
        )
        self.action_index += 1
        return final_action

    def act(self, obs: dict) -> torch.Tensor:
        """Main action function."""
        
        # Extract task_id from observations
        if "task_id" in obs:
            new_task_id = int(obs["task_id"][0])
            self._handle_task_change(new_task_id)
        
        if self.config.control_mode == "rolling_inpainting":
            current_action = self._next_rolling_inpainting_action(obs)
        elif self.config.control_mode == "receding_horizon":
            current_action = self._next_receding_horizon_action(obs)
        elif self.config.control_mode == "temporal_ensemble":
            current_action = self._next_temporal_ensemble_action(obs)
        elif self.config.control_mode == "receding_temporal":
            current_action = self._next_receding_temporal_action(obs)
        elif self.config.control_mode == "hybrid_rolling_temporal":
            current_action = self._next_hybrid_rolling_temporal_action(obs)
        else:
            raise RuntimeError(f"Unexpected control_mode={self.config.control_mode!r}")

        self.step_count += 1
        
        # Log progress every 100 steps
        if self.step_count % 100 == 0:
            if self.task_id is not None:
                stage_status = f"{self.current_stage}/{TASK_NUM_STAGES[self.task_id]-1}"
            else:
                stage_status = f"{self.current_stage}/unknown"
            logger.info(
                f"📊 Step {self.step_count} | Task: {self.task_id} | "
                f"Stage: {stage_status} | Predictions: {self.prediction_count}"
            )
        
        # Convert to torch tensor
        action_tensor = torch.from_numpy(current_action).float()
        if len(action_tensor) > 23:
            action_tensor = action_tensor[:23]
        
        return action_tensor
