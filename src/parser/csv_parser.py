"""
CSV 데이터 파싱 모듈
레거시 코드의 data_loader.py 로직을 개선하여 구현
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CSVParser:
    """CSV 파일 파싱 클래스"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.column_mapping = {}
        self.metadata = {}
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        CSV 파일 로드 (여러 인코딩 시도)
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 원본 데이터
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: CSV 파일 형식이 예상과 다른 경우
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 여러 인코딩 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        
        for encoding in encodings:
            try:
                # 텍스트 파일로 읽어서 직접 파싱
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                
                # 각 행을 쉼표로 분할
                data_rows = []
                max_cols = 0
                
                for line in lines:
                    row = line.strip().split(',')
                    data_rows.append(row)
                    max_cols = max(max_cols, len(row))
                
                # 모든 행의 컬럼 수를 맞춤
                for row in data_rows:
                    while len(row) < max_cols:
                        row.append('')
                
                # DataFrame 생성
                raw_data = pd.DataFrame(data_rows)
                logger.info(f"CSV 파일 로드 완료 (인코딩: {encoding}): {file_path}")
                logger.info(f"데이터 크기: {raw_data.shape}")
                
                self.raw_data = raw_data
                return raw_data
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"인코딩 {encoding}으로 로드 실패: {str(e)}")
                continue
        
        raise ValueError("지원되는 인코딩으로 파일을 로드할 수 없습니다.")
    
    def parse_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        17행 또는 23행에서 컬럼 이름 추출
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            Dict[str, str]: 원본이름:정리된이름 형태의 딕셔너리
        """
        column_mapping = {}
        
        # 17행과 23행 확인 (0-based index: 16, 22)
        row_17 = df.iloc[16] if len(df) > 16 else None
        row_23 = df.iloc[22] if len(df) > 22 else None
        
        # 17행에 'In1'이 있으면 23행 사용
        use_row_23 = False
        if row_17 is not None:
            for value in row_17:
                if pd.notna(value) and 'In1' in str(value):
                    use_row_23 = True
                    break
        
        # 컬럼 이름 결정
        target_row = row_23 if use_row_23 else row_17
        
        if target_row is None:
            raise ValueError("컬럼 이름을 찾을 수 없습니다.")
        
        # 컬럼 이름 정리
        max_cols = len(df.columns)
        
        for col_idx in range(max_cols):
            if col_idx == 0:
                # 첫 번째 컬럼은 사용하지 않음
                column_mapping['unused_col_0'] = 'unused_col_0'
            elif col_idx == 1:
                # 두 번째 컬럼은 time으로 고정
                column_mapping['time'] = 'time(s)'
            else:
                # 17행에서 컬럼 이름 확인
                col_name_17 = None
                if row_17 is not None and col_idx < len(row_17):
                    col_name_17 = row_17.iloc[col_idx]
                
                # 'In1'이면 23행 확인
                if pd.notna(col_name_17) and str(col_name_17).strip() == 'In1':
                    if row_23 is not None and col_idx < len(row_23):
                        col_name_23 = str(row_23.iloc[col_idx])
                        clean_name = self._clean_column_name(col_name_23)
                        column_mapping[col_name_23] = clean_name
                    else:
                        column_mapping[f'col_{col_idx}'] = f'col_{col_idx}'
                
                # 17행에 유효한 이름이 있으면 사용
                elif pd.notna(col_name_17) and str(col_name_17).strip():
                    original_name = str(col_name_17)
                    clean_name = original_name.strip()
                    column_mapping[original_name] = clean_name
                else:
                    column_mapping[f'col_{col_idx}'] = f'col_{col_idx}'
        
        self.column_mapping = column_mapping
        logger.debug(f"컬럼 이름 파싱 완료: {len(column_mapping)}개 컬럼")
        
        return column_mapping
    
    def _clean_column_name(self, name: str) -> str:
        """
        컬럼 이름 정리
        
        Args:
            name: 원본 컬럼 이름
            
        Returns:
            str: 정리된 컬럼 이름
        """
        # 파일 경로인 경우 마지막 부분만 사용
        if '/' in name:
            name = name.split('/')[-1]
        
        # 특수문자 제거 및 공백 처리
        clean_name = name.strip().replace(' ', '_').replace('\\', '_')
        clean_name = clean_name.replace('(', '').replace(')', '')
        clean_name = clean_name.replace('.', '_')
        clean_name = clean_name.replace(':', '_')
        
        if not clean_name or len(clean_name) < 2:
            clean_name = 'unknown_col'
        
        return clean_name
    
    def preprocess_data(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """
        컬럼 이름 정리 및 데이터 타입 변환
        
        Args:
            df: 원본 데이터프레임
            column_map: 컬럼 이름 매핑
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        # 데이터 시작 행 (29행, 0-based index: 28)
        data_start_row = 28
        
        if len(df) <= data_start_row:
            raise ValueError(f"데이터 시작 행({data_start_row + 1}행)을 찾을 수 없습니다.")
        
        # 데이터 추출
        df_processed = df.iloc[data_start_row:].copy()
        
        # 컬럼 이름 변경
        new_columns = []
        for col_idx in range(len(df_processed.columns)):
            if col_idx == 0:
                new_columns.append('unused_col_0')
            elif col_idx == 1:
                new_columns.append('time(s)')
            else:
                # 원본 컬럼 이름 찾기
                mapped_name = self._find_mapped_column_name(df, col_idx, column_map)
                new_columns.append(mapped_name if mapped_name else f'col_{col_idx}')
        
        df_processed.columns = new_columns
        
        # 사용하지 않는 첫 번째 컬럼 제거
        if 'unused_col_0' in df_processed.columns:
            df_processed = df_processed.drop('unused_col_0', axis=1)
        
        # 데이터 타입 변환
        for col in df_processed.columns:
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            except Exception as e:
                logger.warning(f"컬럼 '{col}' 변환 실패: {str(e)}")
        
        # NaN 값이 있는 행 제거
        df_processed = df_processed.dropna(subset=['time(s)'])
        
        # 인덱스 재설정
        df_processed = df_processed.reset_index(drop=True)
        
        self.processed_data = df_processed
        logger.info(f"데이터 전처리 완료: {df_processed.shape}")
        
        return df_processed
    
    def _find_mapped_column_name(self, df: pd.DataFrame, col_idx: int, 
                                  column_map: Dict[str, str]) -> Optional[str]:
        """
        컬럼 인덱스에 해당하는 매핑된 이름 찾기
        
        Args:
            df: 원본 데이터프레임
            col_idx: 컬럼 인덱스
            column_map: 컬럼 매핑
            
        Returns:
            Optional[str]: 매핑된 컬럼 이름 또는 None
        """
        # 17행과 23행 확인
        if len(df) > 16 and col_idx < len(df.iloc[16]):
            col_name_17 = str(df.iloc[16, col_idx])
            
            # 17행이 'In1'이면 23행 확인
            if col_name_17.strip() == 'In1' and len(df) > 22 and col_idx < len(df.iloc[22]):
                col_name_23 = str(df.iloc[22, col_idx])
                for orig_key, clean_name in column_map.items():
                    if orig_key == col_name_23:
                        return clean_name
            else:
                # 17행에서 찾기
                for orig_key, clean_name in column_map.items():
                    if orig_key == col_name_17:
                        return clean_name
        
        return None
    
    def parse(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        CSV 파일 전체 파싱 (로드 + 컬럼 파싱 + 전처리)
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (전처리된 데이터, 메타데이터)
        """
        # 1. CSV 로드
        raw_data = self.load_csv(file_path)
        
        # 2. 컬럼 이름 파싱
        column_mapping = self.parse_column_names(raw_data)
        
        # 3. 데이터 전처리
        processed_data = self.preprocess_data(raw_data, column_mapping)
        
        # 4. 메타데이터 생성
        self.metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'raw_shape': raw_data.shape,
            'processed_shape': processed_data.shape,
            'columns': list(processed_data.columns),
            'time_range': (
                float(processed_data['time(s)'].min()),
                float(processed_data['time(s)'].max())
            ),
            'num_samples': len(processed_data)
        }
        
        return processed_data, self.metadata
    
    def get_processed_data(self) -> pd.DataFrame:
        """전처리된 데이터 반환"""
        return self.processed_data
    
    def get_metadata(self) -> Dict:
        """메타데이터 반환"""
        return self.metadata
    
    def get_column_mapping(self) -> Dict[str, str]:
        """컬럼 매핑 반환"""
        return self.column_mapping


# 사용 예시
if __name__ == "__main__":
    import sys
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 파일 경로
    test_file = "../../data/Arm_Single/A-in-40-H-S.csv"
    
    if os.path.exists(test_file):
        parser = CSVParser()
        
        try:
            data, metadata = parser.parse(test_file)
            
            print("\n=== 메타데이터 ===")
            for key, value in metadata.items():
                print(f"{key}: {value}")
            
            print("\n=== 데이터 샘플 (첫 5행) ===")
            print(data.head())
            
            print("\n=== 데이터 통계 ===")
            print(data.describe())
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
    else:
        print(f"테스트 파일을 찾을 수 없습니다: {test_file}")

