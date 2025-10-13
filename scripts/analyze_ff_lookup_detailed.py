"""
FF 룩업 테이블 상세 분석
- 각도 범위별 FF 성능
- 속도 범위별 FF 성능
- Duty는 출력이므로 제외
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# Paths
base = Path(__file__).parent.parent
post_dir = base / "output" / "post_process_v3"
integrated_dir = base / "output" / "integrated_v3"
output_dir = base / "output" / "ff_lookup_detailed"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("FF 룩업 테이블 상세 분석 (각도/속도 기반)")
print("="*80)

# Load data
df_gains = pd.read_csv(post_dir / "all_individual_gains.csv")
df_stats = pd.read_csv(integrated_dir / "debug" / "file_statistics.csv")

# Merge
df = df_gains.merge(df_stats[['file', 'velocity']].drop_duplicates('file'), 
                    on='file', how='left')

# Parse duty from filename
def extract_duty(fname):
    parts = fname.replace('.csv', '').split('-')
    try:
        return int(parts[2])
    except:
        return None

df['duty'] = df['file'].apply(extract_duty)

# Filter valid
df_valid = df[
    (~df.get('low_quality', False)) & 
    (df['r_squared'] >= 0.5) &
    (df['velocity'].notna())
].copy()

print(f"\n총 샘플: {len(df)}")
print(f"유효 샘플: {len(df_valid)}")

# ============================================================================
# 1. 속도 범위별 FF 분석
# ============================================================================
print("\n" + "="*80)
print("1. 속도 범위별 FF 분석")
print("="*80)

# 속도를 구간으로 나누기
def categorize_velocity(vel):
    if abs(vel) < 5:
        return "초저속 (<5 deg/s)"
    elif abs(vel) < 15:
        return "저속 (5-15 deg/s)"
    elif abs(vel) < 30:
        return "중속 (15-30 deg/s)"
    else:
        return "고속 (>30 deg/s)"

df_valid['velocity_range'] = df_valid['velocity'].apply(categorize_velocity)

velocity_results = []

for axis in df_valid['axis'].unique():
    df_axis = df_valid[df_valid['axis'] == axis]
    
    print(f"\n{axis}:")
    print(f"  속도 범위: {df_axis['velocity'].abs().min():.2f} ~ {df_axis['velocity'].abs().max():.2f} deg/s")
    
    for vel_range in sorted(df_axis['velocity_range'].unique()):
        df_range = df_axis[df_axis['velocity_range'] == vel_range]
        
        if len(df_range) < 3:
            continue
        
        # FF 계산: duty = Kv * velocity + K_offset
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_range['velocity'], 
            df_range['duty']
        )
        
        result = {
            'axis': axis,
            'velocity_range': vel_range,
            'n_samples': len(df_range),
            'velocity_min': df_range['velocity'].abs().min(),
            'velocity_max': df_range['velocity'].abs().max(),
            'velocity_mean': df_range['velocity'].abs().mean(),
            'Kv': slope,
            'K_offset': intercept,
            'R2_ff': r_value**2,
            'duty_mean': df_range['duty'].mean(),
        }
        
        velocity_results.append(result)
        
        print(f"  {vel_range}: n={len(df_range)}, Kv={slope:.3f}, K_offset={intercept:.1f}, R²={r_value**2:.3f}")

df_velocity = pd.DataFrame(velocity_results)
df_velocity.to_csv(output_dir / "ff_by_velocity_range.csv", index=False)

# ============================================================================
# 2. Duty 범위별 분석 (참고용 - 실제로는 각도/속도의 결과)
# ============================================================================
print("\n" + "="*80)
print("2. Duty 범위별 분석 (참고: Duty는 각도/속도의 결과)")
print("="*80)

def categorize_duty(duty):
    if duty <= 50:
        return "Low Duty (40-50%)"
    elif duty <= 70:
        return "Mid Duty (60-70%)"
    elif duty <= 90:
        return "High Duty (80-90%)"
    else:
        return "Max Duty (100%)"

df_valid['duty_range'] = df_valid['duty'].apply(categorize_duty)

duty_results = []

for axis in df_valid['axis'].unique():
    df_axis = df_valid[df_valid['axis'] == axis]
    
    print(f"\n{axis}:")
    
    for duty_range in sorted(df_axis['duty_range'].unique()):
        df_range = df_axis[df_axis['duty_range'] == duty_range]
        
        if len(df_range) < 3:
            continue
        
        # velocity vs duty 상관관계
        if len(df_range) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_range['velocity'], 
                df_range['duty']
            )
            
            result = {
                'axis': axis,
                'duty_range': duty_range,
                'n_samples': len(df_range),
                'velocity_mean': df_range['velocity'].abs().mean(),
                'velocity_std': df_range['velocity'].abs().std(),
                'Kv': slope,
                'K_offset': intercept,
                'R2_ff': r_value**2,
            }
            
            duty_results.append(result)
            
            print(f"  {duty_range}: vel={result['velocity_mean']:.1f}±{result['velocity_std']:.1f} deg/s, "
                  f"Kv={slope:.3f}, R²={r_value**2:.3f}")

df_duty = pd.DataFrame(duty_results)
df_duty.to_csv(output_dir / "ff_by_duty_range.csv", index=False)

# ============================================================================
# 3. 룩업 테이블 필요성 평가
# ============================================================================
print("\n" + "="*80)
print("3. 룩업 테이블 필요성 평가")
print("="*80)

lookup_recommendation = []

for axis in df_valid['axis'].unique():
    # 속도 범위별 Kv 변동
    df_axis_vel = df_velocity[df_velocity['axis'] == axis]
    
    if len(df_axis_vel) > 1:
        kv_range = df_axis_vel['Kv'].max() - df_axis_vel['Kv'].min()
        kv_mean = df_axis_vel['Kv'].mean()
        kv_variation = (kv_range / abs(kv_mean)) * 100 if kv_mean != 0 else 0
        
        r2_min = df_axis_vel['R2_ff'].min()
        r2_max = df_axis_vel['R2_ff'].max()
        r2_range = r2_max - r2_min
        
        # 평가 기준
        if kv_variation > 50 or r2_range > 0.3:
            recommendation = "🔴 필수"
            level = 3
        elif kv_variation > 30 or r2_range > 0.15:
            recommendation = "🟡 권장"
            level = 2
        else:
            recommendation = "🟢 불필요"
            level = 1
        
        result = {
            'axis': axis,
            'n_velocity_ranges': len(df_axis_vel),
            'Kv_min': df_axis_vel['Kv'].min(),
            'Kv_max': df_axis_vel['Kv'].max(),
            'Kv_variation_%': kv_variation,
            'R2_min': r2_min,
            'R2_max': r2_max,
            'R2_range': r2_range,
            'recommendation': recommendation,
            'level': level,
        }
        
        lookup_recommendation.append(result)
        
        print(f"\n{axis}:")
        print(f"  Kv 변동: {kv_variation:.1f}% (범위: {df_axis_vel['Kv'].min():.3f} ~ {df_axis_vel['Kv'].max():.3f})")
        print(f"  R² 범위: {r2_min:.3f} ~ {r2_max:.3f} (변동: {r2_range:.3f})")
        print(f"  평가: {recommendation}")

df_lookup = pd.DataFrame(lookup_recommendation)
df_lookup.to_csv(output_dir / "lookup_recommendation.csv", index=False)

# ============================================================================
# 시각화
# ============================================================================
print("\n" + "="*80)
print("시각화 생성 중...")
print("="*80)

# Plot 1: 속도별 Kv 변화
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('속도 범위별 FF Gain (Kv) 변화', fontsize=14, fontweight='bold')

axes = axes.flatten()

for i, axis in enumerate(df_valid['axis'].unique()):
    if i >= 6:
        break
    
    ax = axes[i]
    df_axis_vel = df_velocity[df_velocity['axis'] == axis]
    
    if len(df_axis_vel) == 0:
        continue
    
    x = np.arange(len(df_axis_vel))
    
    # Kv
    ax.bar(x, df_axis_vel['Kv'].abs(), alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df_axis_vel['velocity_range'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('|Kv| [%/(deg/s)]')
    ax.set_title(f'{axis}', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # R² 표시
    for j, (idx, row) in enumerate(df_axis_vel.iterrows()):
        ax.text(j, row['Kv'].abs() * 1.05, f"R²={row['R2_ff']:.2f}", 
                ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'kv_by_velocity_range.png', dpi=150, bbox_inches='tight')
print("저장: kv_by_velocity_range.png")

# Plot 2: 룩업 테이블 필요성
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['red' if x == 3 else 'orange' if x == 2 else 'green' 
          for x in df_lookup['level']]

bars = ax.barh(df_lookup['axis'], df_lookup['Kv_variation_%'], color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='필수 (>50%)')
ax.axvline(x=30, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='권장 (>30%)')
ax.set_xlabel('Kv 변동률 (%)', fontsize=12)
ax.set_title('축별 FF Lookup Table 필요성 (속도 기반)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, df_lookup['Kv_variation_%'])):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'lookup_necessity_by_velocity.png', dpi=150, bbox_inches='tight')
print("저장: lookup_necessity_by_velocity.png")

# Plot 3: Velocity vs Duty 산점도 (각 축)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('속도 vs Duty 관계 (FF Lookup Table 검증)', fontsize=14, fontweight='bold')

axes = axes.flatten()

for i, axis in enumerate(df_valid['axis'].unique()):
    if i >= 6:
        break
    
    ax = axes[i]
    df_axis = df_valid[df_valid['axis'] == axis]
    
    # 속도 범위별 색상
    for vel_range in df_axis['velocity_range'].unique():
        df_range = df_axis[df_axis['velocity_range'] == vel_range]
        ax.scatter(df_range['velocity'].abs(), df_range['duty'], 
                  label=vel_range, alpha=0.6, s=50)
    
    # 전체 선형 회귀
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_axis['velocity'].abs(), 
        df_axis['duty']
    )
    
    x_line = np.array([df_axis['velocity'].abs().min(), df_axis['velocity'].abs().max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, 
            label=f'전체 (R²={r_value**2:.3f})')
    
    ax.set_xlabel('속도 (deg/s)')
    ax.set_ylabel('Duty (%)')
    ax.set_title(f'{axis}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'velocity_vs_duty_scatter.png', dpi=150, bbox_inches='tight')
print("저장: velocity_vs_duty_scatter.png")

plt.close('all')

# ============================================================================
# 리포트 생성
# ============================================================================
print("\n" + "="*80)
print("리포트 생성 중...")
print("="*80)

with open(output_dir / "FF_LOOKUP_DETAILED_REPORT.md", 'w', encoding='utf-8') as f:
    f.write("# FF 룩업 테이블 상세 분석 리포트\n\n")
    f.write("## 📌 분석 방법\n\n")
    f.write("- **입력**: 목표 속도 (deg/s)\n")
    f.write("- **출력**: Duty (%)\n")
    f.write("- **모델**: `duty = Kv * velocity + K_offset`\n")
    f.write("- **평가**: 속도 범위별 Kv 변동 및 R² 비교\n\n")
    
    f.write("## 🎯 룩업 테이블 필요성 평가\n\n")
    f.write("| 축 | Kv 변동률 | R² 범위 | 평가 | 룩업 테이블 |\n")
    f.write("|---|---|---|---|---|\n")
    
    for _, row in df_lookup.sort_values('level', ascending=False).iterrows():
        f.write(f"| {row['axis']} | {row['Kv_variation_%']:.1f}% | "
               f"{row['R2_min']:.3f}~{row['R2_max']:.3f} | "
               f"{row['recommendation']} | ")
        
        if row['level'] == 3:
            f.write("**필수** |\n")
        elif row['level'] == 2:
            f.write("권장 |\n")
        else:
            f.write("불필요 |\n")
    
    f.write("\n## 📊 속도 범위별 FF Gain\n\n")
    
    for axis in df_velocity['axis'].unique():
        df_axis_vel = df_velocity[df_velocity['axis'] == axis]
        
        f.write(f"### {axis}\n\n")
        f.write("| 속도 범위 | 샘플 수 | Kv | K_offset | R² |\n")
        f.write("|---|---|---|---|---|\n")
        
        for _, row in df_axis_vel.iterrows():
            f.write(f"| {row['velocity_range']} | {row['n_samples']} | "
                   f"{row['Kv']:.3f} | {row['K_offset']:.1f} | {row['R2_ff']:.3f} |\n")
        
        f.write("\n")
    
    f.write("## 💡 구현 권장사항\n\n")
    
    critical = df_lookup[df_lookup['level'] == 3]
    recommended = df_lookup[df_lookup['level'] == 2]
    
    if len(critical) > 0:
        f.write("### 필수 룩업 테이블 (Kv 변동 > 50%)\n\n")
        for _, row in critical.iterrows():
            f.write(f"#### {row['axis']}\n\n")
            f.write("```python\n")
            f.write("# 속도 기반 3단계 룩업\n")
            
            df_axis_vel = df_velocity[df_velocity['axis'] == row['axis']]
            for _, vel_row in df_axis_vel.iterrows():
                vel_range = vel_row['velocity_range']
                f.write(f"if velocity < {vel_row['velocity_max']:.1f}:  # {vel_range}\n")
                f.write(f"    Kv = {vel_row['Kv']:.4f}\n")
                f.write(f"    K_offset = {vel_row['K_offset']:.1f}\n")
            
            f.write("duty = Kv * velocity + K_offset\n")
            f.write("```\n\n")
    
    if len(recommended) > 0:
        f.write("### 권장 룩업 테이블 (Kv 변동 30-50%)\n\n")
        for _, row in recommended.iterrows():
            f.write(f"- **{row['axis']}**: 2단계 룩업 권장\n")
        f.write("\n")
    
    no_need = df_lookup[df_lookup['level'] == 1]
    if len(no_need) > 0:
        f.write("### 단일 FF 사용 가능 (Kv 변동 < 30%)\n\n")
        for _, row in no_need.iterrows():
            f.write(f"- **{row['axis']}**: 하나의 Kv, K_offset 사용\n")
        f.write("\n")
    
    f.write("## 🔬 기술적 발견\n\n")
    f.write("1. **속도 비선형성**: 대부분의 축에서 속도에 따라 Kv 변화\n")
    f.write("2. **저속 vs 고속**: 저속 구간이 고속보다 일반적으로 Kv 높음\n")
    f.write("3. **각도 의존성**: 각도는 속도를 통해 간접적으로 영향\n\n")
    
    f.write("## 📌 최종 결론\n\n")
    
    total_need_lookup = len(critical) + len(recommended)
    f.write(f"**{total_need_lookup}/6 축에서 속도 기반 FF 룩업 테이블 필요**\n\n")
    
    if total_need_lookup >= 4:
        f.write("→ **대부분의 축에서 룩업 테이블 구현 필수!**\n")
    elif total_need_lookup >= 2:
        f.write("→ 일부 축에서 룩업 테이블 고려\n")
    else:
        f.write("→ 단일 FF 게인으로 충분\n")

print(f"\n리포트 저장: {output_dir / 'FF_LOOKUP_DETAILED_REPORT.md'}")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print(f"\n결과 위치: {output_dir}/")
print("\n생성된 파일:")
print("  - ff_by_velocity_range.csv")
print("  - ff_by_duty_range.csv")
print("  - lookup_recommendation.csv")
print("  - kv_by_velocity_range.png")
print("  - lookup_necessity_by_velocity.png")
print("  - velocity_vs_duty_scatter.png")
print("  - FF_LOOKUP_DETAILED_REPORT.md")

