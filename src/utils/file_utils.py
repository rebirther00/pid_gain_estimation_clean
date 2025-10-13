"""
파일 유틸리티 모듈
"""

import os
import glob
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    YAML 파일 로드
    
    Args:
        file_path: YAML 파일 경로
        
    Returns:
        Dict: YAML 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    YAML 파일 저장
    
    Args:
        data: 저장할 데이터
        file_path: YAML 파일 경로
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_pickle(file_path: str) -> Any:
    """
    Pickle 파일 로드
    
    Args:
        file_path: Pickle 파일 경로
        
    Returns:
        Any: 로드된 데이터
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, file_path: str) -> None:
    """
    Pickle 파일 저장
    
    Args:
        data: 저장할 데이터
        file_path: Pickle 파일 경로
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def find_csv_files(data_dir: str, pattern: str = '*.csv') -> List[str]:
    """
    디렉토리에서 CSV 파일 찾기
    
    Args:
        data_dir: 데이터 디렉토리 경로
        pattern: 파일 패턴
        
    Returns:
        List[str]: CSV 파일 경로 리스트
    """
    csv_files = []
    
    # data_dir가 절대 경로가 아니면 현재 디렉토리 기준으로 변환
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.getcwd(), data_dir)
    
    # 디렉토리 존재 확인
    if not os.path.exists(data_dir):
        return csv_files
    
    # 재귀적으로 CSV 파일 찾기
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_result.csv'):
                csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files)


def create_directory(dir_path: str) -> None:
    """
    디렉토리 생성 (존재하지 않는 경우)
    
    Args:
        dir_path: 생성할 디렉토리 경로
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    파일 정보 반환
    
    Args:
        file_path: 파일 경로
        
    Returns:
        Dict: 파일 정보
    """
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'size': stat.st_size,
        'modified': stat.st_mtime
    }

