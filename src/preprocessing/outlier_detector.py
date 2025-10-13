"""
이상치 탐지 모듈
IQR, Z-score 방법으로 이상치 탐지 및 제거
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
import logging

logger = logging.getLogger(__name__)


class OutlierDetector:
    """이상치 탐지 클래스"""
    
    def __init__(self, method: str = 'iqr', threshold: float = None):
        """
        Args:
            method: 탐지 방법 ('iqr', 'zscore', 'velocity_based')
            threshold: 임계값 (None이면 기본값 사용)
        """
        self.method = method
        
        # 기본 임계값 설정
        if threshold is None:
            if method == 'iqr':
                self.threshold = 1.5  # IQR 배수
            elif method == 'zscore':
                self.threshold = 3.0  # 표준편차 배수
            elif method == 'velocity_based':
                self.threshold = 5.0  # 속도 급변화 임계값 (배수)
            else:
                self.threshold = 1.5
        else:
            self.threshold = threshold
    
    def detect(self, data: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상치 탐지
        
        Args:
            data: 입력 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (정상 데이터 인덱스, 이상치 인덱스)
        """
        # Series를 numpy 배열로 변환
        if isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = data
        
        if self.method == 'iqr':
            return self._detect_iqr(data_array)
        elif self.method == 'zscore':
            return self._detect_zscore(data_array)
        elif self.method == 'velocity_based':
            return self._detect_velocity_based(data_array)
        else:
            logger.warning(f"알 수 없는 방법: {self.method}, IQR 사용")
            return self._detect_iqr(data_array)
    
    def _detect_iqr(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        IQR 방법으로 이상치 탐지
        
        Args:
            data: 입력 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (정상 인덱스, 이상치 인덱스)
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        
        normal_idx = np.where((data >= lower_bound) & (data <= upper_bound))[0]
        outlier_idx = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        logger.debug(
            f"IQR 이상치 탐지: {len(outlier_idx)}개 "
            f"(범위: {lower_bound:.2f} ~ {upper_bound:.2f})"
        )
        
        return normal_idx, outlier_idx
    
    def _detect_zscore(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z-score 방법으로 이상치 탐지
        
        Args:
            data: 입력 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (정상 인덱스, 이상치 인덱스)
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            logger.warning("표준편차가 0입니다. 이상치 없음으로 처리")
            return np.arange(len(data)), np.array([])
        
        z_scores = np.abs((data - mean) / std)
        
        normal_idx = np.where(z_scores <= self.threshold)[0]
        outlier_idx = np.where(z_scores > self.threshold)[0]
        
        logger.debug(f"Z-score 이상치 탐지: {len(outlier_idx)}개")
        
        return normal_idx, outlier_idx
    
    def _detect_velocity_based(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        속도 기반 이상치 탐지 (급격한 변화 탐지)
        
        Args:
            data: 입력 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (정상 인덱스, 이상치 인덱스)
        """
        # 차분 계산
        diff = np.diff(data, prepend=data[0])
        
        # 차분의 평균과 표준편차
        mean_diff = np.mean(np.abs(diff))
        std_diff = np.std(diff)
        
        if std_diff == 0:
            logger.warning("차분 표준편차가 0입니다. 이상치 없음으로 처리")
            return np.arange(len(data)), np.array([])
        
        # 임계값 계산
        threshold = mean_diff + self.threshold * std_diff
        
        # 급격한 변화 탐지
        outlier_idx = np.where(np.abs(diff) > threshold)[0]
        normal_idx = np.where(np.abs(diff) <= threshold)[0]
        
        logger.debug(f"속도 기반 이상치 탐지: {len(outlier_idx)}개")
        
        return normal_idx, outlier_idx
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str],
                       inplace: bool = False) -> pd.DataFrame:
        """
        데이터프레임에서 이상치 제거
        
        Args:
            df: 데이터프레임
            columns: 이상치 탐지할 컬럼 리스트
            inplace: 원본 수정 여부
            
        Returns:
            pd.DataFrame: 이상치가 제거된 데이터프레임
        """
        if not inplace:
            df = df.copy()
        
        # 각 컬럼별로 이상치 탐지
        all_outlier_idx = set()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"컬럼 '{col}'을 찾을 수 없습니다.")
                continue
            
            _, outlier_idx = self.detect(df[col])
            all_outlier_idx.update(outlier_idx)
            
            logger.info(f"컬럼 '{col}': {len(outlier_idx)}개 이상치 탐지")
        
        # 이상치 인덱스를 제외한 데이터 반환
        if all_outlier_idx:
            normal_idx = list(set(range(len(df))) - all_outlier_idx)
            df_filtered = df.iloc[sorted(normal_idx)].reset_index(drop=True)
            
            logger.info(
                f"총 {len(all_outlier_idx)}개 이상치 제거 "
                f"({len(df)} -> {len(df_filtered)})"
            )
            
            return df_filtered
        else:
            logger.info("이상치가 발견되지 않았습니다.")
            return df
    
    def mark_outliers(self, df: pd.DataFrame,
                     columns: List[str],
                     outlier_col: str = 'is_outlier') -> pd.DataFrame:
        """
        이상치를 제거하지 않고 표시만 함
        
        Args:
            df: 데이터프레임
            columns: 이상치 탐지할 컬럼 리스트
            outlier_col: 이상치 표시 컬럼명
            
        Returns:
            pd.DataFrame: 이상치가 표시된 데이터프레임
        """
        df_marked = df.copy()
        df_marked[outlier_col] = False
        
        # 각 컬럼별로 이상치 탐지
        all_outlier_idx = set()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"컬럼 '{col}'을 찾을 수 없습니다.")
                continue
            
            _, outlier_idx = self.detect(df[col])
            all_outlier_idx.update(outlier_idx)
        
        # 이상치 표시
        if all_outlier_idx:
            df_marked.loc[list(all_outlier_idx), outlier_col] = True
            logger.info(f"총 {len(all_outlier_idx)}개 이상치 표시")
        else:
            logger.info("이상치가 발견되지 않았습니다.")
        
        return df_marked


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성 (정상 데이터 + 이상치)
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, 1000)
    
    # 이상치 추가 (매우 큰 값 또는 작은 값)
    outlier_indices = np.random.choice(1000, 50, replace=False)
    noisy_data = normal_data.copy()
    noisy_data[outlier_indices] = np.random.choice([0, 100], 50)
    
    # 이상치 탐지 테스트
    methods = [
        ('iqr', 1.5),
        ('zscore', 3.0)
    ]
    
    print("=== 이상치 탐지기 테스트 ===")
    print(f"실제 이상치 개수: {len(outlier_indices)}")
    
    for method, threshold in methods:
        detector = OutlierDetector(method=method, threshold=threshold)
        normal_idx, detected_outlier_idx = detector.detect(noisy_data)
        
        # 정확도 계산
        true_positives = len(set(detected_outlier_idx) & set(outlier_indices))
        precision = true_positives / len(detected_outlier_idx) if len(detected_outlier_idx) > 0 else 0
        recall = true_positives / len(outlier_indices)
        
        print(f"\n{method} (threshold={threshold}):")
        print(f"  탐지된 이상치: {len(detected_outlier_idx)}개")
        print(f"  정밀도 (Precision): {precision:.2f}")
        print(f"  재현율 (Recall): {recall:.2f}")

