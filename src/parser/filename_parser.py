"""
파일명 파싱 모듈
파일명 규칙: (축이름)-(축방향)-(duty값)-(H/L)-(S/C).csv
예시: A-in-40-H-S.csv, B-up-70-L-C.csv
"""

import os
import re
from typing import Dict, Optional
from pathlib import Path


class FilenameParser:
    """파일명 파싱 클래스"""
    
    # 파일명 패턴: (축)-(방향)-(duty)-(부하)-(모드).csv
    FILENAME_PATTERN = r'^([AB])-(in|out|up|dn)-(\d+)-(H|L)-(S|C)\.csv$'
    
    # 축 이름 매핑
    AXIS_MAP = {
        'A': 'Arm/Bucket',  # Arm 폴더에는 Arm과 Bucket 데이터 모두 포함
        'B': 'Boom'
    }
    
    # 방향 매핑
    DIRECTION_MAP = {
        'in': 'In',
        'out': 'Out',
        'up': 'Up',
        'dn': 'Down'
    }
    
    # 부하 매핑
    LOAD_MAP = {
        'H': 'High',
        'L': 'Low'
    }
    
    # 모드 매핑
    MODE_MAP = {
        'S': 'Single',
        'C': 'Couple'
    }
    
    def __init__(self):
        self.pattern = re.compile(self.FILENAME_PATTERN)
    
    def parse(self, filename: str) -> Optional[Dict[str, str]]:
        """
        파일명 파싱
        
        Args:
            filename: CSV 파일명 또는 전체 경로
            
        Returns:
            Dict: 파싱된 메타데이터 또는 None
            {
                'axis': 'A' or 'B',
                'axis_name': 'Arm/Bucket' or 'Boom',
                'direction': 'in', 'out', 'up', 'dn',
                'direction_name': 'In', 'Out', 'Up', 'Down',
                'duty': '40', '50', ..., '100',
                'load': 'H' or 'L',
                'load_name': 'High' or 'Low',
                'mode': 'S' or 'C',
                'mode_name': 'Single' or 'Couple',
                'filename': 원본 파일명,
                'folder': 폴더명 (경로에서 추출)
            }
        """
        # 파일명만 추출
        basename = os.path.basename(filename)
        
        # 패턴 매칭
        match = self.pattern.match(basename)
        
        if not match:
            return None
        
        axis, direction, duty, load, mode = match.groups()
        
        # 메타데이터 딕셔너리 생성
        metadata = {
            'axis': axis,
            'axis_name': self.AXIS_MAP.get(axis, 'Unknown'),
            'direction': direction,
            'direction_name': self.DIRECTION_MAP.get(direction, 'Unknown'),
            'duty': duty,
            'load': load,
            'load_name': self.LOAD_MAP.get(load, 'Unknown'),
            'mode': mode,
            'mode_name': self.MODE_MAP.get(mode, 'Unknown'),
            'filename': basename
        }
        
        # 전체 경로에서 폴더명 추출
        if os.path.dirname(filename):
            folder_name = os.path.basename(os.path.dirname(filename))
            metadata['folder'] = folder_name
        else:
            metadata['folder'] = 'Unknown'
        
        return metadata
    
    def validate(self, filename: str) -> bool:
        """
        파일명 유효성 검증
        
        Args:
            filename: CSV 파일명
            
        Returns:
            bool: 유효하면 True
        """
        basename = os.path.basename(filename)
        return bool(self.pattern.match(basename))
    
    def get_axis_angle_column(self, metadata: Dict[str, str]) -> str:
        """
        축과 폴더에 따른 각도 컬럼명 반환
        
        Args:
            metadata: 파싱된 메타데이터
            
        Returns:
            str: 각도 컬럼명 ('Boom_ang', 'Arm_ang', 'Bkt_ang')
        """
        axis = metadata.get('axis')
        folder = metadata.get('folder', '')
        
        if axis == 'B':
            # Boom과 Bucket을 폴더명으로 구분
            if 'Bucket' in folder:
                return 'Bkt_ang'
            else:
                return 'Boom_ang'
        elif axis == 'A':
            # Arm만 있음 (Bucket은 axis가 'B')
            return 'Arm_ang'
        
        return 'Unknown'
    
    def get_duty_column(self, metadata: Dict[str, str]) -> str:
        """
        duty 컬럼명 반환
        
        Args:
            metadata: 파싱된 메타데이터
            
        Returns:
            str: duty 컬럼명
        """
        axis = metadata.get('axis')
        direction = metadata.get('direction')
        folder = metadata.get('folder', '')
        
        if axis == 'B':
            # Boom과 Bucket을 폴더명으로 구분
            if 'Bucket' in folder:
                if direction == 'in':
                    return 'Bkt_in_duty'
                elif direction == 'out':
                    return 'Bkt_out_duty'
            else:
                # Boom
                if direction == 'up':
                    return 'Boom_up_duty'
                elif direction == 'dn':
                    return 'Boom_dn_duty'
        elif axis == 'A':
            # Arm만 있음 (Bucket은 axis가 'B')
            if direction == 'in':
                return 'Arm_in_duty'
            elif direction == 'out':
                return 'Arm_out_duty'
        
        return 'Unknown'
    
    def format_metadata(self, metadata: Dict[str, str]) -> str:
        """
        메타데이터를 읽기 쉬운 문자열로 포맷
        
        Args:
            metadata: 파싱된 메타데이터
            
        Returns:
            str: 포맷된 문자열
        """
        if not metadata:
            return "Invalid filename"
        
        return (
            f"Axis: {metadata['axis_name']}, "
            f"Direction: {metadata['direction_name']}, "
            f"Duty: {metadata['duty']}%, "
            f"Load: {metadata['load_name']}, "
            f"Mode: {metadata['mode_name']}"
        )


# 사용 예시
if __name__ == "__main__":
    parser = FilenameParser()
    
    # 테스트 파일명들
    test_files = [
        "A-in-40-H-S.csv",
        "A-out-100-L-C.csv",
        "B-up-70-H-S.csv",
        "B-dn-50-L-C.csv",
        "data/Arm_Single/A-in-40-H-S.csv",
        "data/Boom_Couple/B-up-70-L-C.csv"
    ]
    
    for filename in test_files:
        print(f"\n파일명: {filename}")
        metadata = parser.parse(filename)
        if metadata:
            print(f"  {parser.format_metadata(metadata)}")
            print(f"  각도 컬럼: {parser.get_axis_angle_column(metadata)}")
            print(f"  Duty 컬럼: {parser.get_duty_column(metadata)}")
        else:
            print("  파싱 실패!")

