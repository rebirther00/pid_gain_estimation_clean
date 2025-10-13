"""
로깅 설정 모듈
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = 'pid_estimation', level: str = 'INFO', log_dir: str = 'logs') -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR)
        log_dir: 로그 파일 저장 디렉토리
        
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 이미 핸들러가 있으면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'analysis_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 로그 저장
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"로거 초기화 완료: {name}")
    logger.info(f"로그 파일: {log_file}")
    
    return logger

