"""
오차 분석 모듈
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """오차 분석 클래스"""
    
    def __init__(self):
        self.error_metrics = {}
    
    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        오차 분석
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            
        Returns:
            Dict: 오차 지표
        """
        residuals = y_true - y_pred
        
        self.error_metrics = {
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals))),
            'max_error': float(np.max(np.abs(residuals))),
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals))
        }
        
        logger.info(f"오차 분석 완료: RMSE={self.error_metrics['rmse']:.4f}")
        
        return self.error_metrics
    
    def get_metrics(self) -> Dict:
        """오차 지표 반환"""
        return self.error_metrics

