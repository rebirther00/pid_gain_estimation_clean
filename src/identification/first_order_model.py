"""
1차 시스템 모델
G(s) = K * e^(-Ls) / (τs + 1)
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FirstOrderModel:
    """1차 시스템 모델 클래스"""
    
    def __init__(self, K: float = 1.0, tau: float = 1.0, delay: float = 0.0):
        """
        Args:
            K: DC gain
            tau: 시정수 (초)
            delay: 시간 지연 (초)
        """
        self.K = K
        self.tau = tau
        self.delay = delay
    
    def step_response(self, time: np.ndarray, u: float = 1.0) -> np.ndarray:
        """
        계단 입력에 대한 응답 계산
        
        Args:
            time: 시간 배열
            u: 계단 입력 크기
            
        Returns:
            np.ndarray: 출력 응답
        """
        y = np.zeros_like(time)
        
        for i, t in enumerate(time):
            if t < self.delay:
                y[i] = 0
            else:
                y[i] = self.K * u * (1 - np.exp(-(t - self.delay) / self.tau))
        
        return y
    
    def get_parameters(self) -> dict:
        """파라미터 반환"""
        return {
            'K': self.K,
            'tau': self.tau,
            'delay': self.delay
        }
    
    def set_parameters(self, K: float, tau: float, delay: float):
        """파라미터 설정"""
        self.K = K
        self.tau = tau
        self.delay = delay
    
    def __str__(self):
        return f"G(s) = {self.K:.3f} * e^(-{self.delay:.3f}s) / ({self.tau:.3f}s + 1)"

