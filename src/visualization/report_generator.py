"""
리포트 생성 모듈
"""

import json
import yaml
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """리포트 생성 클래스"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_gains_json(self, gains: Dict, filename: str = 'gains.json'):
        """
        게인을 JSON 파일로 저장
        
        Args:
            gains: 게인 딕셔너리
            filename: 파일명
        """
        filepath = self.output_dir / 'gains' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(gains, f, indent=2, ensure_ascii=False)
        
        logger.info(f"게인 저장 완료: {filepath}")
    
    def save_gains_yaml(self, gains: Dict, filename: str = 'gains.yaml'):
        """
        게인을 YAML 파일로 저장
        
        Args:
            gains: 게인 딕셔너리
            filename: 파일명
        """
        filepath = self.output_dir / 'gains' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(gains, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"게인 저장 완료: {filepath}")
    
    def generate_markdown_report(self, data: Dict, filename: str = 'analysis_report.md'):
        """
        Markdown 리포트 생성
        
        Args:
            data: 리포트 데이터
            filename: 파일명
        """
        filepath = self.output_dir / 'reports' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# PID Gain Estimation Report\n\n")
            f.write(f"## Analysis Date: {data.get('date', 'N/A')}\n\n")
            
            # 게인 정보
            if 'gains' in data:
                f.write("## Estimated Gains\n\n")
                for axis, axis_data in data['gains'].items():
                    f.write(f"### {axis}\n\n")
                    for key, value in axis_data.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
            
            # 시스템 식별 결과
            if 'system_id' in data:
                f.write("## System Identification\n\n")
                for key, value in data['system_id'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
        
        logger.info(f"리포트 생성 완료: {filepath}")

