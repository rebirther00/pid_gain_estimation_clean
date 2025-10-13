"""
IMC (Internal Model Control) PID 튜닝 방법
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class IMCTuner:
    """IMC 튜닝 클래스"""
    
    def __init__(self, lambda_factor: float = 1.5):
        """
        Args:
            lambda_factor: 폐루프 시정수 비율 (λ = factor * τ)
                          1~3: 보수적, 0.3~1: 공격적
        """
        self.lambda_factor = lambda_factor
        self.pid_gains = None
    
    def tune(self, K: float, tau: float, lambda_c: float = None) -> Dict[str, float]:
        """
        IMC 방법으로 PID 게인 계산
        
        Args:
            K: DC gain
            tau: 시정수 (초)
            lambda_c: 폐루프 시정수 (None이면 lambda_factor * tau 사용)
            
        Returns:
            Dict: PID 게인 {'Kp', 'Ki', 'Kd'}
        """
        if K == 0:
            logger.warning("K가 0입니다. 기본 게인 반환")
            return {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.0}
        
        # 폐루프 시정수 계산
        if lambda_c is None:
            lambda_c = self.lambda_factor * tau
        
        # IMC 공식 (1차 시스템)
        Kp = tau / (K * lambda_c)
        Ki = 1 / (K * lambda_c)
        Kd = 0  # 1차 시스템에서는 D 게인 불필요
        
        self.pid_gains = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
        
        logger.info(
            f"IMC: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} "
            f"(λ={lambda_c:.3f}s)"
        )
        
        return self.pid_gains
    
    def get_gains(self) -> Dict[str, float]:
        """계산된 PID 게인 반환"""
        return self.pid_gains

