#!/usr/bin/env python3
"""
굴착기 PID/FF 제어기 구현 예제 (ROS2 Python)

날짜: 2025-10-14
모델링: 속도 기반 (Kv = v/duty)
YAML 파일: excavator_pid_ff_gains.yaml
"""

import rclcpp
from rclcpp.node import Node
import yaml
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PIDGains:
    """PID 게인"""
    kp: float
    ki: float
    kd: float = 0.0


@dataclass
class FFGains:
    """FF 게인"""
    kv: float
    k_offset: float


@dataclass
class ControlState:
    """제어 상태"""
    integral: float = 0.0
    prev_error: float = 0.0


class ExcavatorController(Node):
    """굴착기 PID/FF 제어기"""
    
    def __init__(self):
        super().__init__('excavator_controller')
        
        # YAML 파일 로드
        self.load_gains_from_yaml('excavator_gains_simple.yaml')
        
        # 제어 파라미터
        self.sample_time = 0.01  # 100Hz
        self.integral_limit = 10.0
        self.use_safe_gains = True  # 초기에는 안전 게인
        
        # 제어 상태
        self.state_arm = ControlState()
        self.state_boom = ControlState()
        self.state_bucket = ControlState()
        
        # 타이머 (100Hz)
        self.timer = self.create_timer(self.sample_time, self.control_loop)
        
        self.get_logger().info(f'Excavator Controller initialized (Safe mode: {self.use_safe_gains})')
    
    def load_gains_from_yaml(self, yaml_file: str):
        """YAML 파일에서 게인 로드"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # PID 게인 선택 (안전 또는 Full)
            pid_key = 'pid_safe' if self.use_safe_gains else 'pid'
            pid_config = config[pid_key]
            
            # PID 게인 로드
            self.pid_arm_in = PIDGains(**pid_config['arm_in'])
            self.pid_arm_out = PIDGains(**pid_config['arm_out'])
            self.pid_boom_up = PIDGains(**pid_config['boom_up'])
            self.pid_boom_down = PIDGains(**pid_config['boom_down'])
            self.pid_bucket_in = PIDGains(**pid_config['bucket_in'])
            self.pid_bucket_out = PIDGains(**pid_config['bucket_out'])
            
            # FF 게인 로드
            ff_config = config['ff']
            self.ff_arm_in = FFGains(**ff_config['arm_in'])
            self.ff_arm_out = FFGains(**ff_config['arm_out'])
            self.ff_boom_up = FFGains(**ff_config['boom_up'])
            self.ff_boom_down = FFGains(**ff_config['boom_down'])
            self.ff_bucket_in = FFGains(**ff_config['bucket_in'])
            self.ff_bucket_out = FFGains(**ff_config['bucket_out'])
            
            self.get_logger().info(f'Gains loaded from {yaml_file}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load gains: {e}')
            # 기본값 설정
            self._load_default_gains()
    
    def _load_default_gains(self):
        """기본 게인 설정 (YAML 로드 실패 시)"""
        # Full PID 게인
        self.pid_arm_in = PIDGains(3.691, 1.846, 0.0)
        self.pid_arm_out = PIDGains(3.597, 1.799, 0.0)
        self.pid_boom_up = PIDGains(11.431, 5.716, 0.0)
        self.pid_boom_down = PIDGains(6.382, 3.191, 0.0)
        self.pid_bucket_in = PIDGains(1.654, 0.827, 0.0)
        self.pid_bucket_out = PIDGains(1.569, 0.784, 0.0)
        
        # 안전 게인 사용 시 Ki 50%
        if self.use_safe_gains:
            self.pid_arm_in.ki = 0.923
            self.pid_arm_out.ki = 0.899
            self.pid_boom_up.ki = 2.858
            self.pid_boom_down.ki = 1.595
            self.pid_bucket_in.ki = 0.414
            self.pid_bucket_out.ki = 0.392
        
        # FF 게인
        self.ff_arm_in = FFGains(2.709, 36.0)
        self.ff_arm_out = FFGains(1.390, 40.2)
        self.ff_boom_up = FFGains(4.374, 35.9)
        self.ff_boom_down = FFGains(3.134, 35.4)
        self.ff_bucket_in = FFGains(6.045, 0.0)  # K_offset=0
        self.ff_bucket_out = FFGains(0.850, 40.6)
    
    def calculate_pid(self, gains: PIDGains, error: float, state: ControlState, dt: float) -> float:
        """PID 제어 계산"""
        # Proportional
        p_term = gains.kp * error
        
        # Integral (with anti-windup)
        state.integral += error * dt
        state.integral = max(min(state.integral, self.integral_limit), -self.integral_limit)
        i_term = gains.ki * state.integral
        
        # Derivative
        derivative = (error - state.prev_error) / dt
        d_term = gains.kd * derivative
        
        state.prev_error = error
        
        return p_term + i_term + d_term
    
    def calculate_ff(self, gains: FFGains, target_velocity: float) -> float:
        """FF 제어 계산"""
        return gains.kv * abs(target_velocity) + gains.k_offset
    
    def control_axis(
        self,
        target_pos: float,
        current_pos: float,
        gains_pos: PIDGains,
        gains_neg: PIDGains,
        ff_pos: FFGains,
        ff_neg: FFGains,
        state: ControlState
    ) -> Tuple[float, bool]:
        """
        축 제어 (PID + FF)
        
        Returns:
            duty (0~100%), valve_direction (True: In/Up, False: Out/Down)
        """
        dt = self.sample_time
        
        # 1. 목표 속도 계산
        error = target_pos - current_pos
        target_velocity = error / dt
        
        # 2. 방향 결정
        is_positive = (error > 0)
        
        # 3. FF 계산
        if is_positive:
            u_ff = self.calculate_ff(ff_pos, target_velocity)
        else:
            u_ff = self.calculate_ff(ff_neg, target_velocity)
        
        # 4. PID 계산
        if is_positive:
            u_pid = self.calculate_pid(gains_pos, error, state, dt)
        else:
            u_pid = self.calculate_pid(gains_neg, abs(error), state, dt)
        
        # 5. 총 제어 입력
        u_total = u_ff + u_pid
        
        # 6. Duty saturation (0~100%)
        duty = max(min(abs(u_total), 100.0), 0.0)
        
        return duty, is_positive
    
    def control_loop(self):
        """제어 루프 (100Hz)"""
        # 예제: 목표 위치 (실제로는 ROS topic에서 받아옴)
        target_arm = 45.0      # deg
        target_boom = 30.0     # deg
        target_bucket = 20.0   # deg
        
        # 현재 위치 (실제로는 센서에서 받아옴)
        current_arm = 40.0     # deg
        current_boom = 25.0    # deg
        current_bucket = 18.0  # deg
        
        # Arm 제어
        duty_arm, valve_arm = self.control_axis(
            target_arm, current_arm,
            self.pid_arm_in, self.pid_arm_out,
            self.ff_arm_in, self.ff_arm_out,
            self.state_arm
        )
        
        # Boom 제어
        duty_boom, valve_boom = self.control_axis(
            target_boom, current_boom,
            self.pid_boom_up, self.pid_boom_down,
            self.ff_boom_up, self.ff_boom_down,
            self.state_boom
        )
        
        # Bucket 제어
        duty_bucket, valve_bucket = self.control_axis(
            target_bucket, current_bucket,
            self.pid_bucket_in, self.pid_bucket_out,
            self.ff_bucket_in, self.ff_bucket_out,
            self.state_bucket
        )
        
        # 로그 출력 (1초마다)
        if not hasattr(self, '_counter'):
            self._counter = 0
        
        self._counter += 1
        if self._counter >= 100:
            self.get_logger().info(
                f'Arm: {duty_arm:.1f}% ({"IN" if valve_arm else "OUT"}), '
                f'Boom: {duty_boom:.1f}% ({"UP" if valve_boom else "DOWN"}), '
                f'Bucket: {duty_bucket:.1f}% ({"IN" if valve_bucket else "OUT"})'
            )
            self._counter = 0
        
        # 실제로는 여기서 밸브 명령 publish
        # self.publish_valve_commands(duty_arm, valve_arm, ...)
    
    def set_ki_ratio(self, ki_ratio: float):
        """
        Ki 비율 조정 (안전 게인 → Full 게인 점진 증가)
        
        Args:
            ki_ratio: 0.5 (50%) ~ 1.0 (100%)
        """
        ki_ratio = max(min(ki_ratio, 1.0), 0.5)
        
        # Full Ki 값
        full_ki = {
            'arm_in': 1.846,
            'arm_out': 1.799,
            'boom_up': 5.716,
            'boom_down': 3.191,
            'bucket_in': 0.827,
            'bucket_out': 0.784
        }
        
        self.pid_arm_in.ki = full_ki['arm_in'] * ki_ratio
        self.pid_arm_out.ki = full_ki['arm_out'] * ki_ratio
        self.pid_boom_up.ki = full_ki['boom_up'] * ki_ratio
        self.pid_boom_down.ki = full_ki['boom_down'] * ki_ratio
        self.pid_bucket_in.ki = full_ki['bucket_in'] * ki_ratio
        self.pid_bucket_out.ki = full_ki['bucket_out'] * ki_ratio
        
        self.get_logger().info(f'Ki ratio updated to {ki_ratio*100:.0f}%')
    
    def reset_integrators(self):
        """적분기 리셋 (비상 정지 또는 큰 오차 시)"""
        self.state_arm.integral = 0.0
        self.state_boom.integral = 0.0
        self.state_bucket.integral = 0.0
        
        self.get_logger().warn('All integrators reset')


def main(args=None):
    rclcpp.init(args=args)
    
    controller = ExcavatorController()
    
    # 예제: Ki 점진 증가 (별도 스레드)
    import threading
    import time
    
    def increase_ki():
        time.sleep(5)
        controller.set_ki_ratio(0.6)
        
        time.sleep(5)
        controller.set_ki_ratio(0.8)
        
        time.sleep(5)
        controller.set_ki_ratio(1.0)
    
    ki_thread = threading.Thread(target=increase_ki, daemon=True)
    ki_thread.start()
    
    try:
        rclcpp.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclcpp.shutdown()


if __name__ == '__main__':
    main()

