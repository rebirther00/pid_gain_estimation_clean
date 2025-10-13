"""
배치 파서 모듈
같은 축/방향의 모든 CSV 파일을 그룹화하여 파싱
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from .filename_parser import FilenameParser
from .csv_parser import CSVParser
from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class BatchParser:
    """배치 파서 클래스 - 같은 축/방향의 모든 파일 그룹화"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.filename_parser = FilenameParser()
        self.csv_parser = CSVParser()
        self.validator = DataValidator()
        
        # 축/방향 조합 정의
        self.axis_direction_combinations = [
            ('A', 'in', 'Arm', 'In'),      # Arm In
            ('A', 'out', 'Arm', 'Out'),    # Arm Out
            ('B', 'up', 'Boom', 'Up'),     # Boom Up
            ('B', 'dn', 'Boom', 'Down'),   # Boom Down
            ('A', 'in', 'Bucket', 'In'),   # Bucket In (Arm 폴더)
            ('A', 'out', 'Bucket', 'Out'), # Bucket Out (Arm 폴더)
        ]
    
    def find_all_csv_files(self) -> List[str]:
        """
        데이터 디렉토리에서 모든 CSV 파일 찾기
        
        Returns:
            List[str]: CSV 파일 경로 리스트
        """
        csv_files = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.csv') and not file.endswith('_result.csv'):
                    csv_files.append(os.path.join(root, file))
        
        logger.info(f"총 {len(csv_files)}개 CSV 파일 발견")
        return sorted(csv_files)
    
    def group_files_by_axis_direction(self) -> Dict[str, List[str]]:
        """
        축/방향별로 파일 그룹화
        
        Returns:
            Dict: {
                'Arm_In': [파일 리스트],
                'Arm_Out': [파일 리스트],
                'Boom_Up': [파일 리스트],
                'Boom_Down': [파일 리스트],
                'Bucket_In': [파일 리스트],
                'Bucket_Out': [파일 리스트]
            }
        """
        all_files = self.find_all_csv_files()
        
        groups = {
            'Arm_In': [],
            'Arm_Out': [],
            'Boom_Up': [],
            'Boom_Down': [],
            'Bucket_In': [],
            'Bucket_Out': []
        }
        
        for csv_file in all_files:
            metadata = self.filename_parser.parse(csv_file)
            
            if not metadata:
                logger.warning(f"파일명 파싱 실패: {os.path.basename(csv_file)}")
                continue
            
            axis = metadata['axis']
            direction = metadata['direction']
            folder = metadata.get('folder', '')
            
            # 폴더명으로 구분 (Boom과 Bucket이 둘 다 'B'로 시작하므로)
            
            # Arm In (A-in, Arm 폴더)
            if axis == 'A' and direction == 'in' and 'Arm' in folder:
                groups['Arm_In'].append(csv_file)
            # Arm Out (A-out, Arm 폴더)
            elif axis == 'A' and direction == 'out' and 'Arm' in folder:
                groups['Arm_Out'].append(csv_file)
            # Boom Up (B-up, Boom 폴더)
            elif axis == 'B' and direction == 'up' and 'Boom' in folder:
                groups['Boom_Up'].append(csv_file)
            # Boom Down (B-dn, Boom 폴더)
            elif axis == 'B' and direction == 'dn' and 'Boom' in folder:
                groups['Boom_Down'].append(csv_file)
            # Bucket In (B-in, Bucket 폴더)
            elif axis == 'B' and direction == 'in' and 'Bucket' in folder:
                groups['Bucket_In'].append(csv_file)
            # Bucket Out (B-out, Bucket 폴더)
            elif axis == 'B' and direction == 'out' and 'Bucket' in folder:
                groups['Bucket_Out'].append(csv_file)
        
        # 그룹별 파일 개수 출력
        logger.info("\n=== 파일 그룹화 결과 ===")
        for group_name, files in groups.items():
            logger.info(f"{group_name}: {len(files)}개 파일")
            if files:
                # duty 값 확인
                duties = set()
                for f in files:
                    meta = self.filename_parser.parse(f)
                    if meta:
                        duties.add(meta['duty'])
                logger.info(f"  Duty 값: {sorted(duties)}")
        
        return groups
    
    def parse_group(self, group_name: str, file_list: List[str]) -> Dict:
        """
        그룹의 모든 파일 파싱 및 데이터 통합
        
        Args:
            group_name: 그룹 이름 (예: 'Arm_In')
            file_list: 파일 경로 리스트
            
        Returns:
            Dict: {
                'group_name': 'Arm_In',
                'files': [파일 정보 리스트],
                'data_by_duty': {
                    40: {'Single': {'High': data, 'Low': data}, 'Couple': ...},
                    50: {...},
                    ...
                }
            }
        """
        logger.info(f"\n[{group_name}] 파싱 시작")
        
        result = {
            'group_name': group_name,
            'files': [],
            'data_by_duty': {}
        }
        
        for csv_file in file_list:
            try:
                # 파일명 파싱
                metadata = self.filename_parser.parse(csv_file)
                if not metadata:
                    continue
                
                duty = int(metadata['duty'])
                load = metadata['load_name']  # 'High' or 'Low'
                mode = metadata['mode_name']  # 'Single' or 'Couple'
                
                # CSV 파싱
                data, csv_meta = self.csv_parser.parse(csv_file)
                
                # 각도 컬럼 결정
                angle_column = self.filename_parser.get_axis_angle_column(metadata)
                duty_column = self.filename_parser.get_duty_column(metadata)
                
                # 데이터 검증 및 필터링
                data = self.validator.filter_valid_data(data, angle_column, margin=3.0)
                data = self.validator.filter_angle_limits(data, angle_column)
                
                # Duty 필터링
                if duty_column in data.columns:
                    data = data[data[duty_column] != 0].copy()
                    data = data.reset_index(drop=True)
                
                # 데이터 구조화
                if duty not in result['data_by_duty']:
                    result['data_by_duty'][duty] = {
                        'Single': {'High': None, 'Low': None},
                        'Couple': {'High': None, 'Low': None}
                    }
                
                result['data_by_duty'][duty][mode][load] = {
                    'file': csv_file,
                    'data': data,
                    'metadata': metadata,
                    'angle_column': angle_column,
                    'duty_column': duty_column
                }
                
                # 파일 정보 추가
                result['files'].append({
                    'file': csv_file,
                    'duty': duty,
                    'load': load,
                    'mode': mode,
                    'samples': len(data)
                })
                
                logger.info(f"  ✓ {os.path.basename(csv_file)}: {len(data)} 샘플")
                
            except Exception as e:
                logger.error(f"  ✗ {os.path.basename(csv_file)}: {str(e)}")
        
        logger.info(f"[{group_name}] 파싱 완료: {len(result['files'])}개 파일")
        
        return result
    
    def parse_all_groups(self) -> Dict[str, Dict]:
        """
        모든 그룹 파싱
        
        Returns:
            Dict: {
                'Arm_In': {...},
                'Arm_Out': {...},
                ...
            }
        """
        groups = self.group_files_by_axis_direction()
        
        all_results = {}
        
        for group_name, file_list in groups.items():
            if not file_list:
                logger.warning(f"[{group_name}] 파일이 없습니다.")
                continue
            
            result = self.parse_group(group_name, file_list)
            all_results[group_name] = result
        
        return all_results


# 사용 예시
if __name__ == "__main__":
    import sys
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 배치 파서 테스트
    batch_parser = BatchParser(data_dir='../../data')
    
    # 파일 그룹화
    groups = batch_parser.group_files_by_axis_direction()
    
    # Arm In 그룹 파싱 테스트
    if groups['Arm_In']:
        result = batch_parser.parse_group('Arm_In', groups['Arm_In'])
        
        print("\n=== Arm In 파싱 결과 ===")
        print(f"총 파일 수: {len(result['files'])}")
        print(f"Duty 값: {sorted(result['data_by_duty'].keys())}")

