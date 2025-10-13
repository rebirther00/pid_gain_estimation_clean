"""
조건별 Gain 비교 분석
1. Single vs Couple (연결 조건)
2. High vs Low (부하 조건)
3. 각도 범위별 FF 성능
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# Paths
base = Path(__file__).parent.parent
post_dir = base / "output" / "post_process_v3"
integrated_dir = base / "output" / "integrated_v3"
output_dir = base / "output" / "condition_analysis"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("조건별 Gain 비교 분석")
print("="*80)

# Load data
gains_file = post_dir / "all_individual_gains.csv"
stats_file = integrated_dir / "debug" / "file_statistics.csv"

df_gains = pd.read_csv(gains_file)
df_stats = pd.read_csv(stats_file)

# Merge velocity from stats
df = df_gains.merge(df_stats[['file', 'velocity']].drop_duplicates('file'), 
                    on='file', how='left')

# 파일명에서 조건 파싱
def parse_conditions(fname):
    # 예: A-in-40-H-S.csv
    parts = fname.replace('.csv', '').split('-')
    
    # Coupling: S=Single, C=Couple
    coupling = 'Single' if '-S.csv' in fname or parts[-1] == 'S' else 'Couple'
    
    # Load: H=High, L=Low
    load_type = 'High' if '-H-' in fname else 'Low'
    
    # Duty
    try:
        duty = int(parts[2])
    except:
        duty = None
    
    return coupling, load_type, duty

df[['coupling', 'load_type', 'duty_parsed']] = df['file'].apply(
    lambda x: pd.Series(parse_conditions(x))
)

# duty가 없으면 parsed 사용
if 'duty' not in df.columns:
    df['duty'] = df['duty_parsed']

# Filter valid samples
df_valid = df[
    (~df.get('low_quality', False)) & 
    (df['r_squared'] >= 0.5)
].copy()

print(f"\n총 샘플: {len(df)}")
print(f"유효 샘플: {len(df_valid)}")

# ============================================================================
# 1. Single vs Couple 비교
# ============================================================================
print("\n" + "="*80)
print("1. Single vs Couple 비교 (연결 조건)")
print("="*80)

comparison_results = []

for axis in df_valid['axis'].unique():
    df_axis = df_valid[df_valid['axis'] == axis]
    
    single = df_axis[df_axis['coupling'] == 'Single']
    couple = df_axis[df_axis['coupling'] == 'Couple']
    
    if len(single) == 0 or len(couple) == 0:
        continue
    
    result = {
        'axis': axis,
        'n_single': len(single),
        'n_couple': len(couple),
        'Kp_single': single['Kp'].median(),
        'Kp_couple': couple['Kp'].median(),
        'Kp_diff_%': (couple['Kp'].median() / single['Kp'].median() - 1) * 100,
        'Ki_single': single['Ki'].median(),
        'Ki_couple': couple['Ki'].median(),
        'Ki_diff_%': (couple['Ki'].median() / single['Ki'].median() - 1) * 100,
        'R2_single': single['r_squared'].mean(),
        'R2_couple': couple['r_squared'].mean(),
    }
    
    comparison_results.append(result)
    
    print(f"\n{axis}:")
    print(f"  Single: n={len(single)}, Kp={result['Kp_single']:.3f}, Ki={result['Ki_single']:.4f}, R²={result['R2_single']:.3f}")
    print(f"  Couple: n={len(couple)}, Kp={result['Kp_couple']:.3f}, Ki={result['Ki_couple']:.4f}, R²={result['R2_couple']:.3f}")
    print(f"  차이:   Kp {result['Kp_diff_%']:+.1f}%, Ki {result['Ki_diff_%']:+.1f}%")

df_comparison = pd.DataFrame(comparison_results)
df_comparison.to_csv(output_dir / "single_vs_couple.csv", index=False)

# ============================================================================
# 2. High vs Low 부하 비교
# ============================================================================
print("\n" + "="*80)
print("2. High vs Low 부하 비교")
print("="*80)

load_results = []

for axis in df_valid['axis'].unique():
    df_axis = df_valid[df_valid['axis'] == axis]
    
    high = df_axis[df_axis['load_type'] == 'High']
    low = df_axis[df_axis['load_type'] == 'Low']
    
    if len(high) == 0 or len(low) == 0:
        continue
    
    result = {
        'axis': axis,
        'n_high': len(high),
        'n_low': len(low),
        'Kp_high': high['Kp'].median(),
        'Kp_low': low['Kp'].median(),
        'Kp_diff_%': (low['Kp'].median() / high['Kp'].median() - 1) * 100,
        'Ki_high': high['Ki'].median(),
        'Ki_low': low['Ki'].median(),
        'Ki_diff_%': (low['Ki'].median() / high['Ki'].median() - 1) * 100,
        'R2_high': high['r_squared'].mean(),
        'R2_low': low['r_squared'].mean(),
    }
    
    load_results.append(result)
    
    print(f"\n{axis}:")
    print(f"  High Load: n={len(high)}, Kp={result['Kp_high']:.3f}, Ki={result['Ki_high']:.4f}, R²={result['R2_high']:.3f}")
    print(f"  Low Load:  n={len(low)}, Kp={result['Kp_low']:.3f}, Ki={result['Ki_low']:.4f}, R²={result['R2_low']:.3f}")
    print(f"  차이:      Kp {result['Kp_diff_%']:+.1f}%, Ki {result['Ki_diff_%']:+.1f}%")

df_load = pd.DataFrame(load_results)
df_load.to_csv(output_dir / "high_vs_low_load.csv", index=False)

# ============================================================================
# 3. 각도 범위별 FF 성능
# ============================================================================
print("\n" + "="*80)
print("3. 각도 범위별 FF 성능 분석")
print("="*80)

# 각도는 파일명에서 duty로 추정 (간접 지표)
# Duty 40-50: 저각도, 60-80: 중각도, 90-100: 고각도
def categorize_by_duty(duty):
    if duty <= 50:
        return "저속 (40-50%)"
    elif duty <= 80:
        return "중속 (60-80%)"
    else:
        return "고속 (90-100%)"

df_valid['angle_range'] = df_valid['duty'].apply(categorize_by_duty)

ff_angle_results = []

for axis in df_valid['axis'].unique():
    df_axis = df_valid[df_valid['axis'] == axis]
    
    print(f"\n{axis}:")
    
    for angle_range in sorted(df_axis['angle_range'].unique()):
        df_range = df_axis[df_axis['angle_range'] == angle_range]
        
        if len(df_range) < 3:
            continue
        
        # FF 성능 계산 (velocity vs duty 상관관계)
        from scipy import stats
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_range['velocity'], 
            df_range['duty']
        )
        
        result = {
            'axis': axis,
            'angle_range': angle_range,
            'n_samples': len(df_range),
            'Kv_est': slope,
            'K_offset_est': intercept,
            'R2_ff': r_value**2,
            'velocity_mean': df_range['velocity'].mean(),
            'velocity_std': df_range['velocity'].std(),
        }
        
        ff_angle_results.append(result)
        
        print(f"  {angle_range}: n={len(df_range)}, Kv={slope:.3f}, R²={r_value**2:.3f}")

df_ff_angle = pd.DataFrame(ff_angle_results)
df_ff_angle.to_csv(output_dir / "ff_by_angle_range.csv", index=False)

# ============================================================================
# 시각화
# ============================================================================
print("\n" + "="*80)
print("시각화 생성 중...")
print("="*80)

# Plot 1: Single vs Couple Kp 비교
if len(df_comparison) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Kp
    ax = axes[0]
    x = np.arange(len(df_comparison))
    width = 0.35
    
    ax.bar(x - width/2, df_comparison['Kp_single'], width, label='Single', alpha=0.8)
    ax.bar(x + width/2, df_comparison['Kp_couple'], width, label='Couple', alpha=0.8)
    ax.set_xlabel('축')
    ax.set_ylabel('Kp')
    ax.set_title('Single vs Couple: Kp 비교', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['axis'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Kp 차이 (%)
    ax = axes[1]
    colors = ['green' if x < 10 else 'orange' if x < 20 else 'red' 
              for x in abs(df_comparison['Kp_diff_%'])]
    ax.barh(df_comparison['axis'], df_comparison['Kp_diff_%'], color=colors, alpha=0.7)
    ax.set_xlabel('Kp 차이 (%)')
    ax.set_title('Couple - Single Kp 차이', fontweight='bold')
    ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='10%')
    ax.axvline(x=-10, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'single_vs_couple_comparison.png', dpi=150, bbox_inches='tight')
    print(f"저장: single_vs_couple_comparison.png")

# Plot 2: High vs Low Load Kp 비교
if len(df_load) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Kp
    ax = axes[0]
    x = np.arange(len(df_load))
    width = 0.35
    
    ax.bar(x - width/2, df_load['Kp_high'], width, label='High Load', alpha=0.8, color='coral')
    ax.bar(x + width/2, df_load['Kp_low'], width, label='Low Load', alpha=0.8, color='skyblue')
    ax.set_xlabel('축')
    ax.set_ylabel('Kp')
    ax.set_title('High vs Low Load: Kp 비교', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_load['axis'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Kp 차이 (%)
    ax = axes[1]
    colors = ['green' if x < 10 else 'orange' if x < 20 else 'red' 
              for x in abs(df_load['Kp_diff_%'])]
    ax.barh(df_load['axis'], df_load['Kp_diff_%'], color=colors, alpha=0.7)
    ax.set_xlabel('Kp 차이 (%)')
    ax.set_title('Low - High Load Kp 차이', fontweight='bold')
    ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='10%')
    ax.axvline(x=-10, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'high_vs_low_load_comparison.png', dpi=150, bbox_inches='tight')
    print(f"저장: high_vs_low_load_comparison.png")

# Plot 3: FF 각도별 성능
if len(df_ff_angle) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('각도 범위별 FF 성능 (Kv & R²)', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, axis in enumerate(df_valid['axis'].unique()):
        if i >= 6:
            break
        
        ax = axes[i]
        df_axis_ff = df_ff_angle[df_ff_angle['axis'] == axis]
        
        if len(df_axis_ff) == 0:
            continue
        
        # Kv
        ax2 = ax.twinx()
        
        x = np.arange(len(df_axis_ff))
        ax.bar(x, df_axis_ff['Kv_est'], alpha=0.6, label='Kv', color='steelblue')
        ax2.plot(x, df_axis_ff['R2_ff'], 'ro-', linewidth=2, markersize=8, label='R²')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_axis_ff['angle_range'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Kv [%/(deg/s)]', color='steelblue')
        ax2.set_ylabel('R² (FF 정확도)', color='red')
        ax.set_title(f'{axis}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        ax2.set_ylim([0, 1])
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ff_by_angle_range.png', dpi=150, bbox_inches='tight')
    print(f"저장: ff_by_angle_range.png")

plt.close('all')

# ============================================================================
# 요약 리포트 생성
# ============================================================================
print("\n" + "="*80)
print("요약 리포트 생성 중...")
print("="*80)

with open(output_dir / "CONDITION_ANALYSIS_REPORT.md", 'w', encoding='utf-8') as f:
    f.write("# 조건별 Gain 비교 분석 리포트\n\n")
    f.write("## 1. Single vs Couple 비교\n\n")
    
    if len(df_comparison) > 0:
        f.write("| 축 | Single Kp | Couple Kp | 차이 | 평가 |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_comparison.iterrows():
            diff = row['Kp_diff_%']
            if abs(diff) < 10:
                eval = "✅ 무시 가능"
            elif abs(diff) < 20:
                eval = "⚠️ 주의 필요"
            else:
                eval = "🔴 유의미한 차이"
            
            f.write(f"| {row['axis']} | {row['Kp_single']:.3f} | {row['Kp_couple']:.3f} | "
                   f"{diff:+.1f}% | {eval} |\n")
        
        f.write("\n### 결론:\n")
        significant = df_comparison[abs(df_comparison['Kp_diff_%']) > 20]
        if len(significant) > 0:
            f.write(f"- **{len(significant)}개 축에서 Single/Couple 차이 20% 이상**\n")
            for _, row in significant.iterrows():
                f.write(f"  - {row['axis']}: {row['Kp_diff_%']:+.1f}%\n")
            f.write("- **별도 게인 테이블 필요!**\n")
        else:
            f.write("- 모든 축에서 Single/Couple 차이 20% 미만\n")
            f.write("- 단일 게인 사용 가능\n")
    
    f.write("\n## 2. High vs Low Load 비교\n\n")
    
    if len(df_load) > 0:
        f.write("| 축 | High Load Kp | Low Load Kp | 차이 | 평가 |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_load.iterrows():
            diff = row['Kp_diff_%']
            if abs(diff) < 10:
                eval = "✅ 무시 가능"
            elif abs(diff) < 20:
                eval = "⚠️ 주의 필요"
            else:
                eval = "🔴 유의미한 차이"
            
            f.write(f"| {row['axis']} | {row['Kp_high']:.3f} | {row['Kp_low']:.3f} | "
                   f"{diff:+.1f}% | {eval} |\n")
        
        f.write("\n### 결론:\n")
        significant = df_load[abs(df_load['Kp_diff_%']) > 20]
        if len(significant) > 0:
            f.write(f"- **{len(significant)}개 축에서 부하 차이 20% 이상**\n")
            for _, row in significant.iterrows():
                f.write(f"  - {row['axis']}: {row['Kp_diff_%']:+.1f}%\n")
            f.write("- **부하별 게인 스케줄링 고려 필요!**\n")
        else:
            f.write("- 모든 축에서 부하 차이 20% 미만\n")
            f.write("- 단일 게인으로 충분\n")
    
    f.write("\n## 3. 각도 범위별 FF 성능\n\n")
    
    if len(df_ff_angle) > 0:
        f.write("### 축별 FF R² 범위:\n\n")
        for axis in df_ff_angle['axis'].unique():
            df_axis_ff = df_ff_angle[df_ff_angle['axis'] == axis]
            r2_min = df_axis_ff['R2_ff'].min()
            r2_max = df_axis_ff['R2_ff'].max()
            r2_range = r2_max - r2_min
            
            f.write(f"- **{axis}**: R² = {r2_min:.3f} ~ {r2_max:.3f} (범위: {r2_range:.3f})\n")
            
            if r2_range > 0.3:
                f.write(f"  - 🔴 **큰 변동! 각도별 Lookup Table 필수**\n")
            elif r2_range > 0.15:
                f.write(f"  - ⚠️ 중간 변동, Lookup Table 고려\n")
            else:
                f.write(f"  - ✅ 작은 변동, 단일 FF 가능\n")
        
        f.write("\n### 최종 권장:\n")
        needs_lookup = df_ff_angle.groupby('axis')['R2_ff'].agg(lambda x: x.max() - x.min())
        critical = needs_lookup[needs_lookup > 0.3]
        
        if len(critical) > 0:
            f.write(f"\n**{len(critical)}개 축에서 각도별 FF Lookup Table 필요:**\n")
            for axis in critical.index:
                f.write(f"- {axis}\n")
        else:
            f.write("\n**모든 축에서 단일 FF 게인 사용 가능**\n")

print(f"\n리포트 저장: {output_dir / 'CONDITION_ANALYSIS_REPORT.md'}")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print(f"\n결과 위치: {output_dir}/")
print("\n생성된 파일:")
print("  - single_vs_couple.csv")
print("  - high_vs_low_load.csv")
print("  - ff_by_angle_range.csv")
print("  - single_vs_couple_comparison.png")
print("  - high_vs_low_load_comparison.png")
print("  - ff_by_angle_range.png")
print("  - CONDITION_ANALYSIS_REPORT.md")

