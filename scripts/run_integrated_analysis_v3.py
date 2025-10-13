"""
통합 분석 실행 스크립트 V3 (디버깅/검증 기능 추가)
- 파일별 상세 로그
- 10개당 1개 샘플 시각화 (PNG)
- 10개당 1개 파싱 데이터 CSV 출력
"""

import sys
import os
from pathlib import Path
import yaml
import logging
import json
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parser.batch_parser import BatchParser
from src.identification.integrated_analyzer_v3 import IntegratedAnalyzerV3
from src.utils.logger import setup_logger


def main():
    """메인 실행 함수"""
    
    # 로거 설정
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger('integrated_analysis_v3', 'INFO', str(log_dir))
    
    logger.info("="*80)
    logger.info("통합 분석 V3 시작 (디버깅/검증 기능)")
    logger.info("="*80)
    
    # 설정 파일 로드
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 디렉토리
    data_dir = project_root / config['data']['raw_dir']
    
    # 배치 파서 초기화
    parser = BatchParser(str(data_dir))
    
    # 1단계: 파일 그룹화
    logger.info("\n[단계 1] 파일 그룹화 및 전처리")
    file_groups = parser.group_files_by_axis_direction()
    
    # 각 그룹 파싱
    parsed_groups = {}
    for group_name, file_list in file_groups.items():
        if file_list:
            parsed_groups[group_name] = parser.parse_group(group_name, file_list)
    
    # 2단계: 각 그룹 분석 (V3 사용)
    logger.info("\n[단계 2] 통합 분석 V3 (디버깅 모드)")
    
    output_dir = project_root / 'output' / 'integrated_v3'
    analyzer = IntegratedAnalyzerV3(str(output_dir))
    
    results = {}
    all_parsed_samples = []
    
    for group_name, parsed_group in parsed_groups.items():
        try:
            result = analyzer.analyze_group(parsed_group)
            results[group_name] = result
            
            # 파싱 샘플 수집
            if 'parsed_samples' in result:
                all_parsed_samples.extend(result['parsed_samples'])
                
        except Exception as e:
            logger.error(f"[{group_name}] 분석 실패: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 3단계: 결과 저장
    logger.info("\n[단계 3] 결과 저장")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 파일 처리 통계 저장
    analyzer.save_file_statistics()
    
    # 파싱 샘플 CSV 저장 (10개당 1개)
    logger.info("\n파싱 샘플 데이터 저장 (10개당 1개)...")
    analyzer.save_parsed_samples(all_parsed_samples)
    
    # JSON 결과 저장
    output_json = output_dir / f'gains_{timestamp}.json'
    json_results = {}
    for group_name, result in results.items():
        json_results[group_name] = {
            'ff_gain': result['ff_gain'],
            'conservative_pid_gains': result['conservative_gains'],
            'all_pid_gains': result['pid_gains'],
            'model_params': result.get('model_params', {}),
            'goodness_of_fit': result.get('goodness_of_fit', {})
        }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"결과 JSON 저장: {output_json}")
    
    # 텍스트 리포트 생성
    output_txt = output_dir / f'report_{timestamp}.txt'
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"통합 분석 V3 결과 리포트 (디버깅/검증)\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for group_name, result in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{group_name}\n")
            f.write(f"{'='*80}\n\n")
            
            # FF 게인
            ff = result['ff_gain']
            f.write(f"1. FF 게인\n")
            f.write(f"   Kv (속도 게인): {ff['kv']:.4f} %/(deg/s)\n")
            f.write(f"   오프셋: {ff['k_offset']:.4f} %\n")
            f.write(f"   R²: {ff['r_squared']:.4f}\n")
            f.write(f"   방법: {ff['method']}\n")
            if 'data_points' in ff:
                f.write(f"   데이터 포인트: {ff['data_points']}개\n")
            f.write(f"\n")
            
            # PID 게인
            cons = result['conservative_gains']
            f.write(f"2. PID 게인 (권장 - 보수적)\n")
            f.write(f"   Kp: {cons['Kp']:.4f}\n")
            f.write(f"   Ki: {cons['Ki']:.4f}\n")
            f.write(f"   Kd: {cons['Kd']:.4f}\n")
            f.write(f"\n")
            
            # 다른 튜닝 방법들
            if result['pid_gains']:
                f.write(f"3. 다른 PID 튜닝 결과\n\n")
                
                for method_name, gains in result['pid_gains'].items():
                    f.write(f"   {method_name}:\n")
                    f.write(f"     Kp: {gains.get('Kp', 0):.4f}\n")
                    f.write(f"     Ki: {gains.get('Ki', 0):.4f}\n")
                    f.write(f"     Kd: {gains.get('Kd', 0):.4f}\n")
                    f.write(f"\n")
            
            # 모델 파라미터
            if result.get('model_params'):
                model = result['model_params']
                f.write(f"4. 시스템 모델\n")
                f.write(f"   K (DC 게인): {model.get('K', 0):.3f}\n")
                f.write(f"   τ (시정수): {model.get('tau', 0):.3f} s\n")
                f.write(f"   L (지연): {model.get('delay', 0):.3f} s\n")
                
                if result.get('goodness_of_fit'):
                    gof = result['goodness_of_fit']
                    f.write(f"   R²: {gof.get('r_squared', 0):.4f}\n")
                    f.write(f"   RMSE: {gof.get('rmse', 0):.4f}\n")
                f.write(f"\n")
    
    logger.info(f"리포트 저장: {output_txt}")
    
    # 콘솔에 요약 출력
    print("\n" + "="*80)
    print("통합 분석 V3 완료")
    print("="*80)
    
    for group_name, result in results.items():
        print(f"\n[{group_name}]")
        print(f"  FF 게인 (Kv): {result['ff_gain']['kv']:.4f} %/(deg/s)")
        cons = result['conservative_gains']
        print(f"  PID 게인 (보수적):")
        print(f"    Kp: {cons['Kp']:.4f}")
        print(f"    Ki: {cons['Ki']:.4f}")
        print(f"    Kd: {cons['Kd']:.4f}")
    
    print(f"\n결과 저장 위치: {output_dir}")
    print(f"  - 파일 통계: {output_dir / 'debug' / 'file_statistics.csv'}")
    print(f"  - 정상 샘플 그래프: {output_dir / 'debug' / 'plots'}/*.png (10개당 1개)")
    print(f"  - 비정상 데이터 그래프: {output_dir / 'debug' / 'plots'}/ABNORMAL_*.png (전부)")
    print(f"  - 정상 샘플 CSV: {output_dir / 'debug' / 'parsed_data'}/*.csv (10개당 1개)")
    print(f"  - 비정상 데이터 CSV: {output_dir / 'debug' / 'parsed_data'}/ABNORMAL_*.csv (전부)")
    print("="*80)
    
    logger.info("="*80)
    logger.info("통합 분석 V3 완료")
    logger.info("="*80)


if __name__ == "__main__":
    main()

