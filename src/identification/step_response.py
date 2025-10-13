"""
계단 응답 분석 모듈
상승 시간, 정착 시간, 오버슈트 등 계산
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class StepResponseAnalyzer:
    """계단 응답 분석 클래스"""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze(self, time: Union[np.ndarray, pd.Series],
               angle: Union[np.ndarray, pd.Series],
               duty: Union[np.ndarray, pd.Series] = None) -> Dict[str, float]:
        """
        계단 응답 특성 분석
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            duty: Duty 데이터 (옵션)
            
        Returns:
            Dict: 계단 응답 특성
        """
        # Series를 numpy 배열로 변환
        if isinstance(time, pd.Series):
            time = time.values
        if isinstance(angle, pd.Series):
            angle = angle.values
        if isinstance(duty, pd.Series):
            duty = duty.values
        
        # 초기값과 최종값
        initial_value = angle[0]
        final_value = angle[-1]
        change = final_value - initial_value
        
        # Duty 시작 시점 찾기
        if duty is not None:
            duty_start_idx = np.where(duty != 0)[0]
            if len(duty_start_idx) > 0:
                start_idx = duty_start_idx[0]
            else:
                start_idx = 0
        else:
            start_idx = 0
        
        start_time = time[start_idx]
        
        # 1. 상승 시간 (10% ~ 90%)
        rise_time = self._calculate_rise_time(
            time[start_idx:], angle[start_idx:], initial_value, change
        )
        
        # 2. 정착 시간 (95% 또는 98%)
        settling_time_95 = self._calculate_settling_time(
            time[start_idx:], angle[start_idx:], final_value, 0.05
        )
        settling_time_98 = self._calculate_settling_time(
            time[start_idx:], angle[start_idx:], final_value, 0.02
        )
        
        # 3. 오버슈트
        overshoot, peak_time = self._calculate_overshoot(
            time[start_idx:], angle[start_idx:], initial_value, final_value
        )
        
        # 4. 시간 지연 (응답 시작까지의 시간)
        delay = self._calculate_delay(
            time[start_idx:], angle[start_idx:], initial_value, change
        )
        
        # 5. 시정수 추정 (63.2% 도달 시간)
        time_constant = self._estimate_time_constant(
            time[start_idx:], angle[start_idx:], initial_value, change
        )
        
        self.metrics = {
            'initial_value': float(initial_value),
            'final_value': float(final_value),
            'change': float(change),
            'start_time': float(start_time),
            'rise_time': float(rise_time),
            'settling_time_95': float(settling_time_95),
            'settling_time_98': float(settling_time_98),
            'overshoot_percent': float(overshoot),
            'peak_time': float(peak_time),
            'delay': float(delay),
            'time_constant': float(time_constant)
        }
        
        logger.info("계단 응답 분석 완료")
        return self.metrics
    
    def _calculate_rise_time(self, time: np.ndarray, angle: np.ndarray,
                            initial: float, change: float) -> float:
        """
        상승 시간 계산 (10% ~ 90%)
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            initial: 초기값
            change: 변화량
            
        Returns:
            float: 상승 시간 (초)
        """
        # 10%와 90% 값
        value_10 = initial + 0.1 * change
        value_90 = initial + 0.9 * change
        
        # 10% 도달 시점
        if change > 0:
            idx_10 = np.where(angle >= value_10)[0]
            idx_90 = np.where(angle >= value_90)[0]
        else:
            idx_10 = np.where(angle <= value_10)[0]
            idx_90 = np.where(angle <= value_90)[0]
        
        if len(idx_10) == 0 or len(idx_90) == 0:
            logger.warning("10% 또는 90% 지점을 찾을 수 없습니다.")
            return 0.0
        
        time_10 = time[idx_10[0]]
        time_90 = time[idx_90[0]]
        
        rise_time = time_90 - time_10
        
        return rise_time
    
    def _calculate_settling_time(self, time: np.ndarray, angle: np.ndarray,
                                 final: float, tolerance: float) -> float:
        """
        정착 시간 계산
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            final: 최종값
            tolerance: 허용 오차 (예: 0.05 = 5%, 0.02 = 2%)
            
        Returns:
            float: 정착 시간 (초)
        """
        # 허용 범위
        upper_bound = final * (1 + tolerance)
        lower_bound = final * (1 - tolerance)
        
        # 허용 범위 내에 있는지 확인
        within_range = (angle >= lower_bound) & (angle <= upper_bound)
        
        # 마지막부터 역으로 확인하여 처음으로 벗어나는 지점 찾기
        for i in range(len(within_range) - 1, -1, -1):
            if not within_range[i]:
                if i < len(time) - 1:
                    settling_time = time[i + 1] - time[0]
                    return settling_time
        
        # 모든 지점이 범위 내에 있으면 0 반환
        return 0.0
    
    def _calculate_overshoot(self, time: np.ndarray, angle: np.ndarray,
                            initial: float, final: float) -> Tuple[float, float]:
        """
        오버슈트 계산
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            initial: 초기값
            final: 최종값
            
        Returns:
            Tuple[float, float]: (오버슈트 퍼센트, 피크 시간)
        """
        change = final - initial
        
        if change > 0:
            # 증가 방향: 최대값 찾기
            peak_idx = np.argmax(angle)
            peak_value = angle[peak_idx]
            
            if peak_value > final:
                overshoot = ((peak_value - final) / abs(change)) * 100
            else:
                overshoot = 0.0
        else:
            # 감소 방향: 최소값 찾기
            peak_idx = np.argmin(angle)
            peak_value = angle[peak_idx]
            
            if peak_value < final:
                overshoot = ((final - peak_value) / abs(change)) * 100
            else:
                overshoot = 0.0
        
        peak_time = time[peak_idx] - time[0]
        
        return overshoot, peak_time
    
    def _calculate_delay(self, time: np.ndarray, angle: np.ndarray,
                        initial: float, change: float) -> float:
        """
        시간 지연 계산 (응답이 시작되는 시점)
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            initial: 초기값
            change: 변화량
            
        Returns:
            float: 시간 지연 (초)
        """
        # 변화량의 1% 도달 시점을 시간 지연으로 정의
        threshold = initial + 0.01 * change
        
        if change > 0:
            idx = np.where(angle >= threshold)[0]
        else:
            idx = np.where(angle <= threshold)[0]
        
        if len(idx) == 0:
            return 0.0
        
        delay = time[idx[0]] - time[0]
        return delay
    
    def _estimate_time_constant(self, time: np.ndarray, angle: np.ndarray,
                                initial: float, change: float) -> float:
        """
        시정수 추정 (63.2% 도달 시간)
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            initial: 초기값
            change: 변화량
            
        Returns:
            float: 시정수 (초)
        """
        # 63.2% 값
        value_632 = initial + 0.632 * change
        
        if change > 0:
            idx = np.where(angle >= value_632)[0]
        else:
            idx = np.where(angle <= value_632)[0]
        
        if len(idx) == 0:
            logger.warning("63.2% 지점을 찾을 수 없습니다.")
            return 1.0
        
        tau = time[idx[0]] - time[0]
        return tau
    
    def get_metrics(self) -> Dict[str, float]:
        """계단 응답 특성 반환"""
        return self.metrics
    
    def print_summary(self):
        """계단 응답 특성 요약 출력"""
        if not self.metrics:
            print("분석이 수행되지 않았습니다.")
            return
        
        print("\n=== 계단 응답 분석 결과 ===")
        print(f"초기값: {self.metrics['initial_value']:.2f}°")
        print(f"최종값: {self.metrics['final_value']:.2f}°")
        print(f"변화량: {self.metrics['change']:.2f}°")
        print(f"\n상승 시간 (10%~90%): {self.metrics['rise_time']:.3f}s")
        print(f"정착 시간 (±5%): {self.metrics['settling_time_95']:.3f}s")
        print(f"정착 시간 (±2%): {self.metrics['settling_time_98']:.3f}s")
        print(f"오버슈트: {self.metrics['overshoot_percent']:.2f}%")
        print(f"피크 시간: {self.metrics['peak_time']:.3f}s")
        print(f"시간 지연: {self.metrics['delay']:.3f}s")
        print(f"시정수 (τ): {self.metrics['time_constant']:.3f}s")


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성 (1차 시스템 계단 응답)
    K = 50.0  # DC gain
    tau = 1.5  # 시정수
    delay = 0.2  # 시간 지연
    
    t = np.linspace(0, 10, 1000)
    
    # 계단 응답 (시간 지연 포함)
    angle = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < delay:
            angle[i] = 0
        else:
            angle[i] = K * (1 - np.exp(-(ti - delay) / tau))
    
    # 노이즈 추가
    noise = np.random.normal(0, 0.5, len(t))
    noisy_angle = angle + noise
    
    # 계단 응답 분석
    analyzer = StepResponseAnalyzer()
    metrics = analyzer.analyze(t, noisy_angle)
    
    analyzer.print_summary()
    
    print(f"\n=== 실제 파라미터 ===")
    print(f"DC gain: {K}")
    print(f"시정수: {tau}s")
    print(f"시간 지연: {delay}s")
    
    print(f"\n=== 추정 오차 ===")
    print(f"시정수 오차: {abs(metrics['time_constant'] - tau):.3f}s")
    print(f"시간 지연 오차: {abs(metrics['delay'] - delay):.3f}s")

