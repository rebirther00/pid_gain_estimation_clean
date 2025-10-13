"""
Ziegler-Nichols PID 튜닝 방법
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ZieglerNichols:
    """Ziegler-Nichols 튜닝 클래스"""
    
    def __init__(self):
        self.pid_gains = None
    
    def tune(self, K: float, tau: float, L: float) -> Dict[str, float]:
        """
        Ziegler-Nichols 방법으로 PID 게인 계산
        
        1차 시스템: G(s) = K * e^(-Ls) / (τs + 1)
        
        Args:
            K: DC gain
            tau: 시정수 (초)
            L: 시간 지연 (초)
            
        Returns:
            Dict: PID 게인 {'Kp', 'Ki', 'Kd'}
        """
        if K == 0 or L == 0:
            logger.warning("K 또는 L이 0입니다. 기본 게인 반환")
            return {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.01}
        
        # Ziegler-Nichols 공식
        Kp = tau / (K * L)
        Ki = Kp / (2 * L)
        Kd = Kp * L / 2
        
        self.pid_gains = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
        
        logger.info(
            f"Ziegler-Nichols: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}"
        )
        
        return self.pid_gains
    
    def get_gains(self) -> Dict[str, float]:
        """계산된 PID 게인 반환"""
        return self.pid_gains

