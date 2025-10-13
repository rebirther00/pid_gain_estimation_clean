"""
정상 상태 탐지 모듈
속도 변화율 기반으로 정상 상태 구간 식별
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SteadyStateDetector:
    """정상 상태 탐지 클래스"""
    
    def __init__(self, threshold: float = 0.01, 
                 min_duration: float = 1.0,
                 percentage: float = 0.8):
        """
        Args:
            threshold: 속도 변화율 임계값 (deg/s²)
            min_duration: 최소 지속 시간 (초)
            percentage: 마지막 N% 구간 체크 (0~1)
        """
        self.threshold = threshold
        self.min_duration = min_duration
        self.percentage = percentage
    
    def detect(self, velocity: Union[np.ndarray, pd.Series],
              time: Union[np.ndarray, pd.Series]) -> Tuple[int, int]:
        """
        정상 상태 구간 탐지
        
        Args:
            velocity: 속도 데이터
            time: 시간 데이터
            
        Returns:
            Tuple[int, int]: (시작 인덱스, 종료 인덱스)
        """
        # Series를 numpy 배열로 변환
        if isinstance(velocity, pd.Series):
            velocity = velocity.values
        if isinstance(time, pd.Series):
            time = time.values
        
        # 속도 변화율 (가속도) 계산
        acceleration = np.gradient(velocity, time)
        
        # 마지막 N% 구간 체크
        start_idx = int(len(velocity) * self.percentage)
        
        # 정상 상태 조건: 가속도의 절대값이 임계값 이하
        is_steady = np.abs(acceleration[start_idx:]) <= self.threshold
        
        if not is_steady.any():
            logger.warning("정상 상태 구간을 찾을 수 없습니다.")
            return start_idx, len(velocity) - 1
        
        # 정상 상태 시작 인덱스
        steady_start_relative = np.where(is_steady)[0][0]
        steady_start = start_idx + steady_start_relative
        
        # 정상 상태 종료 인덱스 (마지막)
        steady_end = len(velocity) - 1
        
        # 최소 지속 시간 확인
        steady_duration = time[steady_end] - time[steady_start]
        
        if steady_duration < self.min_duration:
            logger.warning(
                f"정상 상태 지속 시간이 짧습니다: {steady_duration:.2f}s "
                f"(최소: {self.min_duration}s)"
            )
        
        logger.debug(
            f"정상 상태 구간: [{steady_start}, {steady_end}] "
            f"(지속시간: {steady_duration:.2f}s)"
        )
        
        return steady_start, steady_end
    
    def detect_by_variance(self, velocity: Union[np.ndarray, pd.Series],
                          time: Union[np.ndarray, pd.Series],
                          variance_threshold: float = 0.1) -> Tuple[int, int]:
        """
        분산 기반 정상 상태 탐지
        
        Args:
            velocity: 속도 데이터
            time: 시간 데이터
            variance_threshold: 분산 임계값
            
        Returns:
            Tuple[int, int]: (시작 인덱스, 종료 인덱스)
        """
        # Series를 numpy 배열로 변환
        if isinstance(velocity, pd.Series):
            velocity = velocity.values
        if isinstance(time, pd.Series):
            time = time.values
        
        # 마지막 N% 구간
        start_idx = int(len(velocity) * self.percentage)
        last_portion = velocity[start_idx:]
        
        # 분산 계산
        variance = np.var(last_portion)
        
        if variance <= variance_threshold:
            logger.debug(
                f"정상 상태 구간: [{start_idx}, {len(velocity)-1}] "
                f"(분산: {variance:.4f})"
            )
            return start_idx, len(velocity) - 1
        else:
            logger.warning(
                f"분산이 임계값보다 큽니다: {variance:.4f} > {variance_threshold}"
            )
            # 더 작은 구간으로 재시도
            start_idx = int(len(velocity) * 0.9)
            last_portion = velocity[start_idx:]
            variance = np.var(last_portion)
            
            logger.debug(
                f"정상 상태 구간 (90%): [{start_idx}, {len(velocity)-1}] "
                f"(분산: {variance:.4f})"
            )
            return start_idx, len(velocity) - 1
    
    def get_steady_state_value(self, velocity: Union[np.ndarray, pd.Series],
                               time: Union[np.ndarray, pd.Series],
                               method: str = 'mean') -> float:
        """
        정상 상태 속도 값 추출
        
        Args:
            velocity: 속도 데이터
            time: 시간 데이터
            method: 추출 방법 ('mean', 'median', 'last')
            
        Returns:
            float: 정상 상태 속도
        """
        # Series를 numpy 배열로 변환
        if isinstance(velocity, pd.Series):
            velocity = velocity.values
        if isinstance(time, pd.Series):
            time = time.values
        
        # 정상 상태 구간 탐지
        start_idx, end_idx = self.detect(velocity, time)
        
        # 정상 상태 구간의 속도
        steady_velocity = velocity[start_idx:end_idx+1]
        
        # 방법에 따라 값 추출
        if method == 'mean':
            value = np.mean(steady_velocity)
        elif method == 'median':
            value = np.median(steady_velocity)
        elif method == 'last':
            value = steady_velocity[-1]
        else:
            logger.warning(f"알 수 없는 방법: {method}, mean 사용")
            value = np.mean(steady_velocity)
        
        logger.info(f"정상 상태 속도: {value:.4f} deg/s (방법: {method})")
        
        return float(value)
    
    def check_steady_state(self, velocity: Union[np.ndarray, pd.Series],
                          time: Union[np.ndarray, pd.Series]) -> bool:
        """
        정상 상태 도달 여부 확인
        
        Args:
            velocity: 속도 데이터
            time: 시간 데이터
            
        Returns:
            bool: 정상 상태 도달 여부
        """
        # Series를 numpy 배열로 변환
        if isinstance(velocity, pd.Series):
            velocity = velocity.values
        if isinstance(time, pd.Series):
            time = time.values
        
        # 정상 상태 구간 탐지
        start_idx, end_idx = self.detect(velocity, time)
        
        # 최소 지속 시간 확인
        steady_duration = time[end_idx] - time[start_idx]
        
        is_steady = steady_duration >= self.min_duration
        
        logger.info(
            f"정상 상태 도달 {'O' if is_steady else 'X'} "
            f"(지속시간: {steady_duration:.2f}s)"
        )
        
        return is_steady
    
    def add_steady_state_info(self, df: pd.DataFrame,
                             velocity_col: str = 'velocity',
                             time_col: str = 'time(s)') -> pd.DataFrame:
        """
        데이터프레임에 정상 상태 정보 추가
        
        Args:
            df: 데이터프레임
            velocity_col: 속도 컬럼명
            time_col: 시간 컬럼명
            
        Returns:
            pd.DataFrame: 정상 상태 정보가 추가된 데이터프레임
        """
        if velocity_col not in df.columns:
            logger.error(f"속도 컬럼 '{velocity_col}'을 찾을 수 없습니다.")
            return df
        
        if time_col not in df.columns:
            logger.error(f"시간 컬럼 '{time_col}'을 찾을 수 없습니다.")
            return df
        
        df_copy = df.copy()
        
        # 정상 상태 구간 탐지
        start_idx, end_idx = self.detect(df[velocity_col], df[time_col])
        
        # 정상 상태 표시 컬럼 추가
        df_copy['is_steady_state'] = False
        df_copy.loc[start_idx:end_idx, 'is_steady_state'] = True
        
        # 정상 상태 속도 추가
        steady_velocity = self.get_steady_state_value(
            df[velocity_col], df[time_col], method='mean'
        )
        df_copy['steady_state_velocity'] = steady_velocity
        
        logger.info("정상 상태 정보 추가 완료")
        
        return df_copy


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성 (계단 응답)
    t = np.linspace(0, 10, 1000)
    
    # 계단 응답: 빠르게 증가 후 정상 상태
    velocity = 50 * (1 - np.exp(-2*t))  # 1차 시스템 응답
    
    # 노이즈 추가
    noise = np.random.normal(0, 0.5, len(t))
    noisy_velocity = velocity + noise
    
    # 정상 상태 탐지 테스트
    detectors = [
        {'threshold': 0.01, 'min_duration': 1.0, 'percentage': 0.8},
        {'threshold': 0.05, 'min_duration': 0.5, 'percentage': 0.7},
    ]
    
    print("=== 정상 상태 탐지기 테스트 ===")
    
    for params in detectors:
        detector = SteadyStateDetector(**params)
        
        start_idx, end_idx = detector.detect(noisy_velocity, t)
        steady_value = detector.get_steady_state_value(noisy_velocity, t, method='mean')
        is_steady = detector.check_steady_state(noisy_velocity, t)
        
        print(f"\n파라미터: {params}")
        print(f"  정상 상태 구간: [{start_idx}, {end_idx}]")
        print(f"  정상 상태 속도: {steady_value:.4f} deg/s")
        print(f"  정상 상태 도달: {'O' if is_steady else 'X'}")
        print(f"  실제 최종 속도: {velocity[-1]:.4f} deg/s (오차: {abs(steady_value - velocity[-1]):.4f})")

