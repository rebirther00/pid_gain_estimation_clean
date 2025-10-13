"""
속도 추정 모듈
각도 데이터로부터 각속도 및 가속도 계산
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
import logging

from ..utils.constants import SAMPLING_RATE

logger = logging.getLogger(__name__)


class VelocityEstimator:
    """속도 추정 클래스"""
    
    def __init__(self, method: str = 'central_diff'):
        """
        Args:
            method: 미분 방법 ('central_diff', 'forward_diff', 'backward_diff')
        """
        self.method = method
    
    def estimate_velocity(self, angle: Union[np.ndarray, pd.Series], 
                         time: Union[np.ndarray, pd.Series],
                         apply_filter: bool = False) -> np.ndarray:
        """
        각속도 계산
        
        Args:
            angle: 각도 데이터
            time: 시간 데이터
            apply_filter: 필터 적용 여부
            
        Returns:
            np.ndarray: 각속도 (deg/s)
        """
        # Series를 numpy 배열로 변환
        if isinstance(angle, pd.Series):
            angle = angle.values
        if isinstance(time, pd.Series):
            time = time.values
        
        # 미분 계산
        if self.method == 'central_diff':
            velocity = self._central_difference(angle, time)
        elif self.method == 'forward_diff':
            velocity = self._forward_difference(angle, time)
        elif self.method == 'backward_diff':
            velocity = self._backward_difference(angle, time)
        else:
            logger.warning(f"알 수 없는 방법: {self.method}, central_diff 사용")
            velocity = self._central_difference(angle, time)
        
        # 필터 적용 (옵션)
        if apply_filter:
            from .noise_filter import NoiseFilter
            noise_filter = NoiseFilter('savgol', window_length=11, polyorder=3)
            velocity = noise_filter.filter(velocity)
        
        logger.debug(f"각속도 계산 완료 (방법: {self.method})")
        return velocity
    
    def estimate_acceleration(self, velocity: Union[np.ndarray, pd.Series],
                             time: Union[np.ndarray, pd.Series],
                             apply_filter: bool = False) -> np.ndarray:
        """
        각가속도 계산
        
        Args:
            velocity: 각속도 데이터
            time: 시간 데이터
            apply_filter: 필터 적용 여부
            
        Returns:
            np.ndarray: 각가속도 (deg/s²)
        """
        # Series를 numpy 배열로 변환
        if isinstance(velocity, pd.Series):
            velocity = velocity.values
        if isinstance(time, pd.Series):
            time = time.values
        
        # 미분 계산
        if self.method == 'central_diff':
            acceleration = self._central_difference(velocity, time)
        elif self.method == 'forward_diff':
            acceleration = self._forward_difference(velocity, time)
        elif self.method == 'backward_diff':
            acceleration = self._backward_difference(velocity, time)
        else:
            acceleration = self._central_difference(velocity, time)
        
        # 필터 적용 (옵션)
        if apply_filter:
            from .noise_filter import NoiseFilter
            noise_filter = NoiseFilter('savgol', window_length=11, polyorder=3)
            acceleration = noise_filter.filter(acceleration)
        
        logger.debug(f"각가속도 계산 완료 (방법: {self.method})")
        return acceleration
    
    def _central_difference(self, data: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        중앙 차분법
        
        Args:
            data: 입력 데이터
            time: 시간 데이터
            
        Returns:
            np.ndarray: 미분 값
        """
        # numpy gradient 사용 (중앙 차분법)
        return np.gradient(data, time)
    
    def _forward_difference(self, data: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        전진 차분법
        
        Args:
            data: 입력 데이터
            time: 시간 데이터
            
        Returns:
            np.ndarray: 미분 값
        """
        # 시간 간격 계산
        dt = np.diff(time, prepend=time[0])
        # 데이터 차분
        dd = np.diff(data, prepend=data[0])
        
        return dd / dt
    
    def _backward_difference(self, data: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        후진 차분법
        
        Args:
            data: 입력 데이터
            time: 시간 데이터
            
        Returns:
            np.ndarray: 미분 값
        """
        # 시간 간격 계산
        dt = np.diff(time, append=time[-1])
        # 데이터 차분
        dd = np.diff(data, append=data[-1])
        
        return dd / dt
    
    def estimate_all(self, angle: Union[np.ndarray, pd.Series],
                    time: Union[np.ndarray, pd.Series],
                    apply_filter: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        각속도와 각가속도 동시 계산
        
        Args:
            angle: 각도 데이터
            time: 시간 데이터
            apply_filter: 필터 적용 여부
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (각속도, 각가속도)
        """
        velocity = self.estimate_velocity(angle, time, apply_filter)
        acceleration = self.estimate_acceleration(velocity, time, apply_filter)
        
        return velocity, acceleration
    
    def add_to_dataframe(self, df: pd.DataFrame, 
                        angle_col: str,
                        time_col: str = 'time(s)',
                        velocity_col: str = 'velocity',
                        acceleration_col: str = 'acceleration',
                        apply_filter: bool = False) -> pd.DataFrame:
        """
        데이터프레임에 속도 및 가속도 컬럼 추가
        
        Args:
            df: 데이터프레임
            angle_col: 각도 컬럼명
            time_col: 시간 컬럼명
            velocity_col: 추가할 속도 컬럼명
            acceleration_col: 추가할 가속도 컬럼명
            apply_filter: 필터 적용 여부
            
        Returns:
            pd.DataFrame: 속도/가속도가 추가된 데이터프레임
        """
        if angle_col not in df.columns:
            logger.error(f"각도 컬럼 '{angle_col}'을 찾을 수 없습니다.")
            return df
        
        if time_col not in df.columns:
            logger.error(f"시간 컬럼 '{time_col}'을 찾을 수 없습니다.")
            return df
        
        df_copy = df.copy()
        
        # 각속도 계산
        velocity = self.estimate_velocity(df[angle_col], df[time_col], apply_filter)
        df_copy[velocity_col] = velocity
        
        # 각가속도 계산
        acceleration = self.estimate_acceleration(velocity, df[time_col], apply_filter)
        df_copy[acceleration_col] = acceleration
        
        logger.info(f"속도 및 가속도 계산 완료: {velocity_col}, {acceleration_col}")
        
        return df_copy


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성 (사인파)
    t = np.linspace(0, 10, 1000)
    angle = 50 * np.sin(2 * np.pi * 0.5 * t)  # 진폭 50도, 주파수 0.5Hz
    
    # 이론적 속도 및 가속도
    theoretical_velocity = 50 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
    theoretical_acceleration = -50 * (2 * np.pi * 0.5)**2 * np.sin(2 * np.pi * 0.5 * t)
    
    # 속도 추정기 테스트
    methods = ['central_diff', 'forward_diff', 'backward_diff']
    
    print("=== 속도 추정기 테스트 ===")
    for method in methods:
        estimator = VelocityEstimator(method=method)
        velocity, acceleration = estimator.estimate_all(angle, t, apply_filter=False)
        
        # RMSE 계산
        velocity_rmse = np.sqrt(np.mean((velocity - theoretical_velocity) ** 2))
        acceleration_rmse = np.sqrt(np.mean((acceleration - theoretical_acceleration) ** 2))
        
        print(f"{method}:")
        print(f"  속도 RMSE: {velocity_rmse:.4f}")
        print(f"  가속도 RMSE: {acceleration_rmse:.4f}")

