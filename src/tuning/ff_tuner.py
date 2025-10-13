"""
피드포워드 게인 추정 모듈
"""

import numpy as np
import pandas as pd
from typing import Union, Dict
import logging

logger = logging.getLogger(__name__)


class FFTuner:
    """피드포워드 게인 추정 클래스"""
    
    def __init__(self):
        self.ff_gain = None
        self.method_used = None
    
    def estimate_from_steady_state(self, steady_velocity: float, 
                                   duty: float) -> float:
        """
        정상 상태 속도로부터 FF 게인 추정
        
        Args:
            steady_velocity: 정상 상태 속도 (deg/s)
            duty: Duty 값 (%)
            
        Returns:
            float: FF 게인 K_ff [%/(deg/s)]
        """
        if steady_velocity == 0:
            logger.warning("정상 상태 속도가 0입니다.")
            return 0.0
        
        ff_gain = duty / steady_velocity
        self.ff_gain = ff_gain
        self.method_used = 'steady_state'
        
        logger.info(f"FF 게인 (정상 상태): {ff_gain:.4f} %/(deg/s)")
        
        return ff_gain
    
    def estimate_from_dc_gain(self, K_dc: float) -> float:
        """
        DC gain으로부터 FF 게인 추정
        
        Args:
            K_dc: 시스템 DC gain (deg/%)
            
        Returns:
            float: FF 게인 K_ff [%/(deg/s)]
        """
        if K_dc == 0:
            logger.warning("DC gain이 0입니다.")
            return 0.0
        
        ff_gain = 1.0 / K_dc
        self.ff_gain = ff_gain
        self.method_used = 'dc_gain'
        
        logger.info(f"FF 게인 (DC gain): {ff_gain:.4f} %/(deg/s)")
        
        return ff_gain
    
    def estimate_all_methods(self, steady_velocity: float, duty: float,
                            K_dc: float) -> Dict[str, float]:
        """
        두 가지 방법으로 FF 게인 추정 및 비교
        
        Args:
            steady_velocity: 정상 상태 속도 (deg/s)
            duty: Duty 값 (%)
            K_dc: DC gain (deg/%)
            
        Returns:
            Dict: 각 방법별 FF 게인
        """
        ff_ss = self.estimate_from_steady_state(steady_velocity, duty)
        ff_dc = self.estimate_from_dc_gain(K_dc)
        
        # 평균값 사용
        ff_avg = (ff_ss + ff_dc) / 2.0
        
        result = {
            'steady_state': ff_ss,
            'dc_gain': ff_dc,
            'average': ff_avg,
            'difference_percent': abs(ff_ss - ff_dc) / ff_avg * 100 if ff_avg != 0 else 0
        }
        
        logger.info(f"FF 게인 평균: {ff_avg:.4f} %/(deg/s) (차이: {result['difference_percent']:.2f}%)")
        
        return result
    
    def get_ff_gain(self) -> float:
        """추정된 FF 게인 반환"""
        return self.ff_gain

