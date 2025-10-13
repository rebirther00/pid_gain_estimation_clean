"""
노이즈 필터링 모듈
Savitzky-Golay, Moving Average, Butterworth 필터 제공
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class NoiseFilter:
    """노이즈 필터 클래스"""
    
    def __init__(self, filter_type: str = 'savgol', **kwargs):
        """
        Args:
            filter_type: 필터 타입 ('savgol', 'moving_average', 'butter')
            **kwargs: 필터별 파라미터
        """
        self.filter_type = filter_type
        self.filter_params = kwargs
    
    def filter(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        데이터 필터링
        
        Args:
            data: 입력 데이터
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        # Series를 numpy 배열로 변환
        if isinstance(data, pd.Series):
            data = data.values
        
        if self.filter_type == 'savgol':
            return self._savgol_filter(data)
        elif self.filter_type == 'moving_average':
            return self._moving_average_filter(data)
        elif self.filter_type == 'butter':
            return self._butterworth_filter(data)
        else:
            logger.warning(f"알 수 없는 필터 타입: {self.filter_type}, 원본 데이터 반환")
            return data
    
    def _savgol_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay 필터
        
        Args:
            data: 입력 데이터
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        window_length = self.filter_params.get('window_length', 11)
        polyorder = self.filter_params.get('polyorder', 3)
        
        # window_length는 홀수여야 함
        if window_length % 2 == 0:
            window_length += 1
        
        # window_length는 데이터 길이보다 작아야 함
        if window_length > len(data):
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        
        # polyorder는 window_length보다 작아야 함
        if polyorder >= window_length:
            polyorder = window_length - 1
        
        try:
            filtered_data = signal.savgol_filter(data, window_length, polyorder)
            logger.debug(f"Savitzky-Golay 필터 적용 (window={window_length}, poly={polyorder})")
            return filtered_data
        except Exception as e:
            logger.error(f"Savitzky-Golay 필터 적용 실패: {str(e)}")
            return data
    
    def _moving_average_filter(self, data: np.ndarray) -> np.ndarray:
        """
        이동 평균 필터
        
        Args:
            data: 입력 데이터
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        window_size = self.filter_params.get('window_size', 5)
        
        # window_size는 데이터 길이보다 작아야 함
        if window_size > len(data):
            window_size = len(data)
        
        window = np.ones(window_size) / window_size
        filtered_data = np.convolve(data, window, mode='same')
        
        logger.debug(f"이동 평균 필터 적용 (window={window_size})")
        return filtered_data
    
    def _butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Butterworth 저역통과 필터
        
        Args:
            data: 입력 데이터
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        cutoff = self.filter_params.get('cutoff', 5.0)  # Hz
        fs = self.filter_params.get('fs', 100.0)  # Hz
        order = self.filter_params.get('order', 4)
        
        # Nyquist 주파수
        nyquist = fs / 2.0
        normal_cutoff = cutoff / nyquist
        
        # 차단 주파수 범위 체크
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.95
            logger.warning(f"차단 주파수가 너무 높습니다. 0.95로 조정")
        
        try:
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            filtered_data = signal.filtfilt(b, a, data)
            logger.debug(f"Butterworth 필터 적용 (cutoff={cutoff}Hz, order={order})")
            return filtered_data
        except Exception as e:
            logger.error(f"Butterworth 필터 적용 실패: {str(e)}")
            return data
    
    def filter_dataframe(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        데이터프레임의 여러 컬럼에 필터 적용
        
        Args:
            df: 데이터프레임
            columns: 필터링할 컬럼 리스트
            
        Returns:
            pd.DataFrame: 필터링된 데이터프레임
        """
        df_filtered = df.copy()
        
        for col in columns:
            if col in df.columns:
                df_filtered[col] = self.filter(df[col])
                logger.info(f"컬럼 '{col}' 필터링 완료")
            else:
                logger.warning(f"컬럼 '{col}'을 찾을 수 없습니다.")
        
        return df_filtered


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성 (노이즈가 추가된 사인파)
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * 1 * t)
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = clean_signal + noise
    
    # 필터 테스트
    filters = [
        ('savgol', {'window_length': 11, 'polyorder': 3}),
        ('moving_average', {'window_size': 5}),
        ('butter', {'cutoff': 5.0, 'fs': 100.0, 'order': 4})
    ]
    
    print("=== 노이즈 필터 테스트 ===")
    for filter_type, params in filters:
        noise_filter = NoiseFilter(filter_type, **params)
        filtered_signal = noise_filter.filter(noisy_signal)
        
        # RMSE 계산
        rmse = np.sqrt(np.mean((filtered_signal - clean_signal) ** 2))
        print(f"{filter_type}: RMSE = {rmse:.4f}")

