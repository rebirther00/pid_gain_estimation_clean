"""
커플링 분석 모듈
Single vs Couple 비교 분석
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CouplingAnalyzer:
    """커플링 분석 클래스"""
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Args:
            significance_threshold: 유의미한 차이 임계값 (5% 이상)
        """
        self.significance_threshold = significance_threshold
        self.analysis_result = {}
    
    def analyze(self, single_params: Dict, couple_params: Dict) -> Dict:
        """
        Single과 Couple 파라미터 비교 분석
        
        Args:
            single_params: Single 모드 파라미터
            couple_params: Couple 모드 파라미터
            
        Returns:
            Dict: 분석 결과
        """
        result = {}
        
        for key in single_params.keys():
            if key in couple_params:
                single_val = single_params[key]
                couple_val = couple_params[key]
                
                # 차이율 계산
                if single_val != 0:
                    diff_pct = abs((couple_val - single_val) / single_val)
                else:
                    diff_pct = 0.0
                
                # 유의미한 차이인지 판단
                is_significant = diff_pct >= self.significance_threshold
                
                result[key] = {
                    'single': single_val,
                    'couple': couple_val,
                    'difference_percent': diff_pct * 100,
                    'is_significant': is_significant
                }
        
        self.analysis_result = result
        
        # 요약
        significant_params = [k for k, v in result.items() if v['is_significant']]
        
        logger.info(f"커플링 분석 완료: {len(significant_params)}개 파라미터에서 유의미한 차이")
        
        return result
    
    def get_coupling_effect(self) -> float:
        """
        평균 커플링 효과 계산
        
        Returns:
            float: 평균 차이율 (%)
        """
        if not self.analysis_result:
            return 0.0
        
        diff_pcts = [v['difference_percent'] for v in self.analysis_result.values()]
        return np.mean(diff_pcts)

