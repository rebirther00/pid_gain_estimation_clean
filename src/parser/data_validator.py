"""
데이터 검증 모듈
각도 범위, 샘플링 레이트, 결측치 등을 검증
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from ..utils.constants import ANGLE_LIMITS, SAMPLING_RATE, VALID_DATA_MARGIN

logger = logging.getLogger(__name__)


class DataValidator:
    """데이터 검증 클래스"""
    
    def __init__(self, angle_limits: Dict[str, Dict[str, float]] = None):
        """
        Args:
            angle_limits: 각도 범위 제한 (None이면 기본값 사용)
        """
        self.angle_limits = angle_limits if angle_limits else ANGLE_LIMITS
        self.validation_results = {}
    
    def validate_all(self, df: pd.DataFrame, angle_column: str) -> Dict[str, any]:
        """
        전체 데이터 검증
        
        Args:
            df: 데이터프레임
            angle_column: 검증할 각도 컬럼명
            
        Returns:
            Dict: 검증 결과
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # 1. 기본 검증
        basic_result = self.validate_basic(df)
        results['info']['basic'] = basic_result
        if not basic_result['is_valid']:
            results['is_valid'] = False
            results['errors'].extend(basic_result['errors'])
        
        # 2. 샘플링 레이트 검증
        sampling_result = self.validate_sampling_rate(df)
        results['info']['sampling'] = sampling_result
        if not sampling_result['is_valid']:
            results['warnings'].append(sampling_result['message'])
        
        # 3. 각도 범위 검증
        angle_result = self.validate_angle_range(df, angle_column)
        results['info']['angle'] = angle_result
        if not angle_result['is_valid']:
            results['warnings'].append(angle_result['message'])
        
        # 4. 결측치 검증
        missing_result = self.validate_missing_data(df)
        results['info']['missing'] = missing_result
        if not missing_result['is_valid']:
            results['warnings'].append(missing_result['message'])
        
        self.validation_results = results
        return results
    
    def validate_basic(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        기본 검증 (필수 컬럼 존재 여부, 데이터 크기 등)
        
        Args:
            df: 데이터프레임
            
        Returns:
            Dict: 검증 결과
        """
        result = {
            'is_valid': True,
            'errors': []
        }
        
        # time(s) 컬럼 존재 확인
        if 'time(s)' not in df.columns:
            result['is_valid'] = False
            result['errors'].append("time(s) 컬럼이 없습니다.")
        
        # 데이터가 비어있는지 확인
        if len(df) == 0:
            result['is_valid'] = False
            result['errors'].append("데이터가 비어있습니다.")
        
        # 최소 데이터 개수 확인 (최소 100개 샘플)
        if len(df) < 100:
            result['is_valid'] = False
            result['errors'].append(f"데이터가 너무 적습니다 (샘플 수: {len(df)})")
        
        return result
    
    def validate_sampling_rate(self, df: pd.DataFrame, 
                               expected_rate: float = SAMPLING_RATE,
                               tolerance: float = 0.002) -> Dict[str, any]:
        """
        샘플링 레이트 검증
        
        Args:
            df: 데이터프레임
            expected_rate: 예상 샘플링 레이트 (초)
            tolerance: 허용 오차
            
        Returns:
            Dict: 검증 결과
        """
        if 'time(s)' not in df.columns or len(df) < 2:
            return {'is_valid': False, 'message': 'time(s) 컬럼이 없거나 데이터가 부족합니다.'}
        
        # 시간 간격 계산
        time_diffs = df['time(s)'].diff().dropna()
        
        # 평균 샘플링 레이트
        avg_sampling_rate = time_diffs.mean()
        
        # 검증
        is_valid = abs(avg_sampling_rate - expected_rate) <= tolerance
        
        result = {
            'is_valid': is_valid,
            'expected': expected_rate,
            'actual': float(avg_sampling_rate),
            'min': float(time_diffs.min()),
            'max': float(time_diffs.max()),
            'std': float(time_diffs.std())
        }
        
        if not is_valid:
            result['message'] = (
                f"샘플링 레이트가 예상과 다릅니다. "
                f"예상: {expected_rate:.4f}s, 실제: {avg_sampling_rate:.4f}s"
            )
        else:
            result['message'] = "샘플링 레이트가 정상입니다."
        
        return result
    
    def validate_angle_range(self, df: pd.DataFrame, angle_column: str) -> Dict[str, any]:
        """
        각도 범위 검증
        
        Args:
            df: 데이터프레임
            angle_column: 각도 컬럼명
            
        Returns:
            Dict: 검증 결과
        """
        if angle_column not in df.columns:
            return {
                'is_valid': False,
                'message': f"각도 컬럼 '{angle_column}'이 존재하지 않습니다."
            }
        
        if angle_column not in self.angle_limits:
            return {
                'is_valid': True,
                'message': f"각도 컬럼 '{angle_column}'에 대한 제한이 설정되지 않았습니다."
            }
        
        limits = self.angle_limits[angle_column]
        angles = df[angle_column]
        
        # 범위 벗어나는 데이터 개수
        out_of_range = ((angles < limits['min']) | (angles > limits['max'])).sum()
        out_of_range_pct = (out_of_range / len(angles)) * 100
        
        # 10% 이상 벗어나면 경고
        is_valid = out_of_range_pct < 10.0
        
        result = {
            'is_valid': is_valid,
            'limits': limits,
            'actual_min': float(angles.min()),
            'actual_max': float(angles.max()),
            'out_of_range_count': int(out_of_range),
            'out_of_range_percentage': float(out_of_range_pct)
        }
        
        if not is_valid:
            result['message'] = (
                f"각도 범위를 벗어나는 데이터가 많습니다. "
                f"({out_of_range}개, {out_of_range_pct:.1f}%)"
            )
        else:
            result['message'] = "각도 범위가 정상입니다."
        
        return result
    
    def validate_missing_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        결측치 검증
        
        Args:
            df: 데이터프레임
            
        Returns:
            Dict: 검증 결과
        """
        total_values = df.size
        missing_values = df.isna().sum().sum()
        missing_pct = (missing_values / total_values) * 100
        
        # 5% 이상 결측치가 있으면 경고
        is_valid = missing_pct < 5.0
        
        # 컬럼별 결측치 개수
        missing_by_column = df.isna().sum()
        missing_cols = missing_by_column[missing_by_column > 0].to_dict()
        
        result = {
            'is_valid': is_valid,
            'total_values': total_values,
            'missing_values': int(missing_values),
            'missing_percentage': float(missing_pct),
            'missing_by_column': {k: int(v) for k, v in missing_cols.items()}
        }
        
        if not is_valid:
            result['message'] = (
                f"결측치가 많습니다. ({missing_values}개, {missing_pct:.1f}%)"
            )
        else:
            result['message'] = "결측치가 정상 범위입니다."
        
        return result
    
    def filter_valid_data(self, df: pd.DataFrame, angle_column: str, 
                         margin: float = VALID_DATA_MARGIN) -> pd.DataFrame:
        """
        유효 데이터 필터링 (최종 각도 - margin도까지만 사용)
        
        Args:
            df: 데이터프레임
            angle_column: 각도 컬럼명
            margin: 마진 (도)
            
        Returns:
            pd.DataFrame: 필터링된 데이터
        """
        if angle_column not in df.columns or len(df) == 0:
            return df
        
        # 최종 각도
        final_angle = df[angle_column].iloc[-1]
        
        # 각도 변화 방향 확인
        initial_angle = df[angle_column].iloc[0]
        direction = 1 if final_angle > initial_angle else -1
        
        # 필터링 조건
        if direction > 0:
            # 증가 방향: 최종 각도 - margin 이하
            threshold = final_angle - margin
            filtered_df = df[df[angle_column] <= threshold].copy()
        else:
            # 감소 방향: 최종 각도 + margin 이상
            threshold = final_angle + margin
            filtered_df = df[df[angle_column] >= threshold].copy()
        
        removed_count = len(df) - len(filtered_df)
        if removed_count > 0:
            logger.info(
                f"유효 데이터 필터링: {removed_count}개 샘플 제거 "
                f"(최종 각도: {final_angle:.2f}°, 임계값: {threshold:.2f}°)"
            )
        
        return filtered_df
    
    def filter_angle_limits(self, df: pd.DataFrame, angle_column: str) -> pd.DataFrame:
        """
        각도 범위 제한 필터링
        
        Args:
            df: 데이터프레임
            angle_column: 각도 컬럼명
            
        Returns:
            pd.DataFrame: 필터링된 데이터
        """
        if angle_column not in df.columns:
            return df
        
        if angle_column not in self.angle_limits:
            logger.warning(f"각도 컬럼 '{angle_column}'에 대한 제한이 설정되지 않았습니다.")
            return df
        
        limits = self.angle_limits[angle_column]
        original_len = len(df)
        
        # 범위 내 데이터만 유지
        filtered_df = df[
            (df[angle_column] >= limits['min']) & 
            (df[angle_column] <= limits['max'])
        ].copy()
        
        removed_count = original_len - len(filtered_df)
        if removed_count > 0:
            logger.info(
                f"각도 범위 필터링: {removed_count}개 샘플 제거 "
                f"(범위: {limits['min']:.1f}° ~ {limits['max']:.1f}°)"
            )
        
        return filtered_df
    
    def get_validation_summary(self) -> str:
        """
        검증 결과 요약 문자열 반환
        
        Returns:
            str: 검증 결과 요약
        """
        if not self.validation_results:
            return "검증이 수행되지 않았습니다."
        
        results = self.validation_results
        summary = []
        
        summary.append("=== 데이터 검증 결과 ===")
        summary.append(f"전체 유효성: {'통과' if results['is_valid'] else '실패'}")
        
        if results['errors']:
            summary.append("\n오류:")
            for error in results['errors']:
                summary.append(f"  - {error}")
        
        if results['warnings']:
            summary.append("\n경고:")
            for warning in results['warnings']:
                summary.append(f"  - {warning}")
        
        return "\n".join(summary)


# 사용 예시
if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    
    from src.parser.csv_parser import CSVParser
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 파일
    test_file = "../../data/Arm_Single/A-in-40-H-S.csv"
    
    if os.path.exists(test_file):
        # 데이터 로드
        parser = CSVParser()
        data, metadata = parser.parse(test_file)
        
        # 검증
        validator = DataValidator()
        validation_result = validator.validate_all(data, 'Arm_ang')
        
        print(validator.get_validation_summary())
        
        # 유효 데이터 필터링
        filtered_data = validator.filter_valid_data(data, 'Arm_ang')
        print(f"\n원본 데이터: {len(data)}개 샘플")
        print(f"필터링 후: {len(filtered_data)}개 샘플")
    else:
        print(f"테스트 파일을 찾을 수 없습니다: {test_file}")

