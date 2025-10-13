"""
모든 샘플의 개별 PID 게인을 하나의 통합 파일로 생성
"""

import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / 'output' / 'post_process_v3'

print("="*80)
print("모든 샘플의 개별 PID 게인 통합")
print("="*80)

# 개별 파일 로드 (all_individual_gains.csv 제외)
csv_files = [f for f in output_dir.glob('*_individual_gains.csv') 
             if f.name != 'all_individual_gains.csv']

if not csv_files:
    print("\n오류: 개별 게인 파일이 없습니다!")
    print(f"위치: {output_dir}")
    print("\n먼저 post_process_v3_results.py를 실행하세요.")
    sys.exit(1)

print(f"\n발견된 파일: {len(csv_files)}개")

# 모든 데이터 로드 및 축 정보 추가
all_data = []

for csv_file in csv_files:
    # 파일명에서 축 이름 추출
    axis_name = csv_file.stem.replace('_individual_gains', '')
    print(f"  - {axis_name}: {csv_file.name}")
    
    df = pd.read_csv(csv_file)
    df.insert(0, 'axis', axis_name)  # 축 정보를 첫 번째 컬럼으로 추가
    all_data.append(df)

# 통합
combined_df = pd.concat(all_data, ignore_index=True)

# 컬럼 순서 재정렬
column_order = [
    'axis',           # 축
    'file',           # 파일명
    'duty',           # Duty
    'mode',           # Single/Couple
    'load',           # High/Low
    'Kp',             # PID 게인
    'Ki',
    'Kd',
    'K',              # 모델 파라미터
    'tau',
    'delay',
    'r_squared',      # 품질 지표
    'rmse',
    'low_quality'     # 품질 플래그
]

# low_quality 컬럼이 없는 경우 대비
if 'low_quality' not in combined_df.columns:
    combined_df['low_quality'] = False

combined_df = combined_df[column_order]

# Duty로 정렬
combined_df = combined_df.sort_values(['axis', 'duty', 'mode', 'load'])

print(f"\n전체 샘플 수: {len(combined_df)}개")
print(f"\n축별 샘플 수:")
for axis in combined_df['axis'].unique():
    count = len(combined_df[combined_df['axis'] == axis])
    print(f"  {axis}: {count}개")

# CSV 저장
output_csv = output_dir / 'all_individual_gains.csv'
combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n[OK] CSV 저장: {output_csv}")

# Excel 저장 (더 보기 좋음)
output_excel = output_dir / 'all_individual_gains.xlsx'
try:
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 전체 시트
        combined_df.to_excel(writer, sheet_name='All_Samples', index=False)
        
        # 축별 시트
        for axis in combined_df['axis'].unique():
            axis_df = combined_df[combined_df['axis'] == axis]
            axis_df.to_excel(writer, sheet_name=axis, index=False)
        
        # 요약 통계 시트
        summary_data = []
        for axis in combined_df['axis'].unique():
            axis_df = combined_df[combined_df['axis'] == axis]
            summary_data.append({
                'Axis': axis,
                'N_samples': len(axis_df),
                'Kp_mean': axis_df['Kp'].mean(),
                'Kp_std': axis_df['Kp'].std(),
                'Kp_median': axis_df['Kp'].median(),
                'Kp_min': axis_df['Kp'].min(),
                'Kp_max': axis_df['Kp'].max(),
                'Ki_mean': axis_df['Ki'].mean(),
                'Ki_std': axis_df['Ki'].std(),
                'Ki_median': axis_df['Ki'].median(),
                'Ki_min': axis_df['Ki'].min(),
                'Ki_max': axis_df['Ki'].max(),
                'R2_mean': axis_df['r_squared'].mean(),
                'R2_min': axis_df['r_squared'].min(),
                'R2_max': axis_df['r_squared'].max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"[OK] Excel 저장: {output_excel}")
    print(f"\n  시트 구성:")
    print(f"    - All_Samples: 전체 데이터 ({len(combined_df)}개)")
    for axis in combined_df['axis'].unique():
        count = len(combined_df[combined_df['axis'] == axis])
        print(f"    - {axis}: {axis} 데이터 ({count}개)")
    print(f"    - Summary_Statistics: 축별 통계 요약")
    
except ImportError:
    print("\n[WARNING] Excel 저장 실패: openpyxl 미설치")
    print("   설치: pip install openpyxl")

# 간단한 통계 출력
print(f"\n{'='*80}")
print("축별 통계 (중앙값)")
print(f"{'='*80}")
print(f"{'축':<15} {'샘플':<8} {'Kp':<12} {'Ki':<12} {'Kd':<12} {'R²':<10}")
print(f"{'-'*80}")

for axis in combined_df['axis'].unique():
    axis_df = combined_df[combined_df['axis'] == axis]
    kp_median = axis_df['Kp'].median()
    ki_median = axis_df['Ki'].median()
    kd_median = axis_df['Kd'].median()
    r2_mean = axis_df['r_squared'].mean()
    
    print(f"{axis:<15} {len(axis_df):<8} {kp_median:<12.6f} {ki_median:<12.6f} {kd_median:<12.6f} {r2_mean:<10.4f}")

print(f"\n{'='*80}")
print("완료!")
print(f"{'='*80}")

