"""
모델 피팅 모듈
1차 시스템 모델 파라미터 추정
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares
from typing import Dict, Tuple
import logging

from .first_order_model import FirstOrderModel
from ..utils.math_utils import calculate_r_squared, calculate_rmse

logger = logging.getLogger(__name__)


class ModelFitter:
    """모델 피팅 클래스"""
    
    def __init__(self):
        self.model = None
        self.fitted_params = None
        self.goodness_of_fit = {}
    
    def fit_first_order(self, time: np.ndarray, angle: np.ndarray,
                       initial_guess: Dict = None) -> FirstOrderModel:
        """
        1차 시스템 모델 피팅
        
        Args:
            time: 시간 데이터
            angle: 각도 데이터
            initial_guess: 초기 추정값 {'K': 1.0, 'tau': 1.0, 'delay': 0.1}
            
        Returns:
            FirstOrderModel: 피팅된 모델
        """
        # 초기 추정값 설정
        if initial_guess is None:
            K_init = angle[-1] - angle[0]  # 최종 변화량
            tau_init = (time[-1] - time[0]) / 3  # 전체 시간의 1/3
            delay_init = 0.1
        else:
            K_init = initial_guess.get('K', 1.0)
            tau_init = initial_guess.get('tau', 1.0)
            delay_init = initial_guess.get('delay', 0.1)
        
        # 피팅 함수 정의
        def first_order_func(t, K, tau, delay):
            y = np.zeros_like(t)
            for i, ti in enumerate(t):
                if ti < delay:
                    y[i] = angle[0]
                else:
                    y[i] = angle[0] + K * (1 - np.exp(-(ti - delay) / tau))
            return y
        
        try:
            # 커브 피팅
            p0 = [K_init, tau_init, delay_init]
            # K는 음수도 허용 (In/Down 방향 대응)
            bounds = ([-np.inf, 0.01, 0], [np.inf, 100, 1.0])  # 파라미터 범위
            
            params, _ = curve_fit(
                first_order_func,
                time,
                angle,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            
            K_fit, tau_fit, delay_fit = params
            
            # 모델 생성
            self.model = FirstOrderModel(K_fit, tau_fit, delay_fit)
            self.fitted_params = {'K': K_fit, 'tau': tau_fit, 'delay': delay_fit}
            
            # 적합도 평가
            y_pred = first_order_func(time, K_fit, tau_fit, delay_fit)
            self._evaluate_fit(angle, y_pred)
            
            logger.info(f"모델 피팅 완료: K={K_fit:.3f}, τ={tau_fit:.3f}s, delay={delay_fit:.3f}s")
            logger.info(f"R²={self.goodness_of_fit['r_squared']:.4f}, RMSE={self.goodness_of_fit['rmse']:.4f}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"모델 피팅 실패: {str(e)}")
            # 기본 모델 반환
            return FirstOrderModel(K_init, tau_init, delay_init)
    
    def _evaluate_fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        적합도 평가
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
        """
        self.goodness_of_fit = {
            'r_squared': calculate_r_squared(y_true, y_pred),
            'rmse': calculate_rmse(y_true, y_pred),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'max_error': np.max(np.abs(y_true - y_pred))
        }
    
    def get_fitted_params(self) -> Dict:
        """피팅된 파라미터 반환"""
        return self.fitted_params
    
    def get_goodness_of_fit(self) -> Dict:
        """적합도 지표 반환"""
        return self.goodness_of_fit
    
    def is_good_fit(self, min_r_squared: float = 0.9, max_rmse: float = 5.0) -> bool:
        """
        적합도 판정
        
        Args:
            min_r_squared: 최소 R² 요구사항
            max_rmse: 최대 RMSE 요구사항
            
        Returns:
            bool: 적합도가 충분한지 여부
        """
        if not self.goodness_of_fit:
            return False
        
        r_squared_ok = self.goodness_of_fit['r_squared'] >= min_r_squared
        rmse_ok = self.goodness_of_fit['rmse'] <= max_rmse
        
        return r_squared_ok and rmse_ok

