"""
수학 유틸리티 모듈
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def calculate_derivative(data: np.ndarray, dt: float, method: str = 'central') -> np.ndarray:
    """
    미분 계산 (속도 또는 가속도)
    
    Args:
        data: 입력 데이터 배열
        dt: 시간 간격
        method: 미분 방법 ('central', 'forward', 'backward')
        
    Returns:
        np.ndarray: 미분 값
    """
    if method == 'central':
        # 중앙 차분법
        derivative = np.gradient(data, dt)
    elif method == 'forward':
        # 전진 차분법
        derivative = np.diff(data, prepend=data[0]) / dt
    elif method == 'backward':
        # 후진 차분법
        derivative = np.diff(data, append=data[-1]) / dt
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return derivative


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    이동 평균 필터
    
    Args:
        data: 입력 데이터
        window_size: 윈도우 크기
        
    Returns:
        np.ndarray: 필터링된 데이터
    """
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')


def savgol_filter(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay 필터
    
    Args:
        data: 입력 데이터
        window_length: 윈도우 길이 (홀수)
        polyorder: 다항식 차수
        
    Returns:
        np.ndarray: 필터링된 데이터
    """
    if window_length % 2 == 0:
        window_length += 1
    
    return signal.savgol_filter(data, window_length, polyorder)


def butterworth_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Butterworth 저역통과 필터
    
    Args:
        data: 입력 데이터
        cutoff: 차단 주파수 (Hz)
        fs: 샘플링 주파수 (Hz)
        order: 필터 차수
        
    Returns:
        np.ndarray: 필터링된 데이터
    """
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    
    # 차단 주파수 범위 체크
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.95
    
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def detect_outliers_iqr(data: np.ndarray, threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    IQR 방법으로 이상치 탐지
    
    Args:
        data: 입력 데이터
        threshold: IQR 배수 임계값
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (정상 데이터 인덱스, 이상치 인덱스)
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    normal_idx = np.where((data >= lower_bound) & (data <= upper_bound))[0]
    outlier_idx = np.where((data < lower_bound) | (data > upper_bound))[0]
    
    return normal_idx, outlier_idx


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-score 방법으로 이상치 탐지
    
    Args:
        data: 입력 데이터
        threshold: Z-score 임계값
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (정상 데이터 인덱스, 이상치 인덱스)
    """
    mean = np.mean(data)
    std = np.std(data)
    
    z_scores = np.abs((data - mean) / std)
    
    normal_idx = np.where(z_scores <= threshold)[0]
    outlier_idx = np.where(z_scores > threshold)[0]
    
    return normal_idx, outlier_idx


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    결정계수 (R²) 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        float: R² 값
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE (Root Mean Square Error) 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        float: RMSE 값
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE (Mean Absolute Error) 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        float: MAE 값
    """
    return np.mean(np.abs(y_true - y_pred))

