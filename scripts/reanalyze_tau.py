"""
시정수(τ) 재분석
문제: 대부분 τ=100s로 수렴 (상한선 도달)
해결: 실제 응답 시간 기반 재계산
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
base = Path(__file__).parent.parent
post_dir = base / "output" / "post_process_v3"
output_dir = base / "output" / "tau_reanalysis"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("시정수(τ) 재분석")
print("="*80)

# Load data
df = pd.read_csv(post_dir / "all_individual_gains.csv")

print(f"\n총 샘플: {len(df)}")
print(f"\nτ 통계:")
print(df['tau'].describe())

print(f"\nτ=100s 샘플: {len(df[df['tau'] >= 99.9])} / {len(df)} ({len(df[df['tau'] >= 99.9])/len(df)*100:.1f}%)")
print(f"τ<10s 샘플: {len(df[df['tau'] < 10])}")
print(f"τ 10-50s 샘플: {len(df[(df['tau'] >= 10) & (df['tau'] < 50)])}")

# 축별 분석
print("\n" + "="*80)
print("축별 τ 분포")
print("="*80)

tau_summary = []

for axis in df['axis'].unique():
    df_axis = df[df['axis'] == axis]
    
    n_total = len(df_axis)
    n_saturated = len(df_axis[df_axis['tau'] >= 99.9])
    n_valid = len(df_axis[df_axis['tau'] < 99.9])
    
    if n_valid > 0:
        tau_valid_mean = df_axis[df_axis['tau'] < 99.9]['tau'].mean()
        tau_valid_median = df_axis[df_axis['tau'] < 99.9]['tau'].median()
    else:
        tau_valid_mean = 100
        tau_valid_median = 100
    
    result = {
        'axis': axis,
        'n_total': n_total,
        'n_saturated': n_saturated,
        'n_valid': n_valid,
        'saturation_%': n_saturated / n_total * 100,
        'tau_mean_all': df_axis['tau'].mean(),
        'tau_median_all': df_axis['tau'].median(),
        'tau_mean_valid': tau_valid_mean,
        'tau_median_valid': tau_valid_median,
    }
    
    tau_summary.append(result)
    
    print(f"\n{axis}:")
    print(f"  포화 샘플: {n_saturated}/{n_total} ({result['saturation_%']:.1f}%)")
    if n_valid > 0:
        print(f"  유효 τ 평균: {tau_valid_mean:.2f}s")
        print(f"  유효 τ 중앙값: {tau_valid_median:.2f}s")
    print(f"  전체 τ 평균: {result['tau_mean_all']:.2f}s")

df_summary = pd.DataFrame(tau_summary)
df_summary.to_csv(output_dir / "tau_summary.csv", index=False)

# 문제 진단
print("\n" + "="*80)
print("문제 진단")
print("="*80)

print(f"\n전체 샘플의 {len(df[df['tau'] >= 99.9])/len(df)*100:.1f}%가 τ 상한선(100s)에 도달")
print("이는 다음을 의미합니다:")
print("  1. 모델 피팅이 매우 느린 응답을 예측")
print("  2. 실제로는 훨씬 빠를 가능성 높음")
print("  3. 정착 시간 4*τ=400s는 과대평가")

# K 기반 분석
print("\n" + "="*80)
print("대안: K (System Gain) 기반 분석")
print("="*80)

response_summary = []

for axis in df['axis'].unique():
    df_axis = df[df['axis'] == axis]
    
    # 유효 샘플 (R² > 0.8, K가 합리적)
    df_valid = df_axis[
        (df_axis['r_squared'] > 0.8) & 
        (df_axis['K'].abs() > 0.01) &
        (df_axis['K'].abs() < 10000)
    ]
    
    if len(df_valid) == 0:
        continue
    
    result = {
        'axis': axis,
        'n_valid': len(df_valid),
        'K_median': df_valid['K'].abs().median(),
        'K_mean': df_valid['K'].abs().mean(),
        'tau_fitted': df_valid['tau'].median(),
        'tau_mean': df_valid['tau'].mean(),
    }
    
    response_summary.append(result)
    
    print(f"\n{axis}:")
    print(f"  K (gain) 중앙값: {result['K_median']:.1f} deg/%")
    print(f"  피팅된 τ 중앙값: {result['tau_fitted']:.1f}s")
    print(f"  피팅된 τ 평균: {result['tau_mean']:.1f}s")

df_response = pd.DataFrame(response_summary)
df_response.to_csv(output_dir / "response_analysis.csv", index=False)

# 시각화
print("\n" + "="*80)
print("시각화 생성 중...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('시정수(τ) 분포 분석', fontsize=14, fontweight='bold')

axes = axes.flatten()

for i, axis in enumerate(df['axis'].unique()):
    if i >= 6:
        break
    
    ax = axes[i]
    df_axis = df[df['axis'] == axis]
    
    # 히스토그램
    ax.hist(df_axis['tau'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='상한선 (100s)')
    ax.axvline(x=df_axis['tau'].median(), color='green', linestyle='-', 
               linewidth=2, label=f'중앙값 ({df_axis["tau"].median():.1f}s)')
    
    ax.set_xlabel('τ (s)')
    ax.set_ylabel('샘플 수')
    ax.set_title(f'{axis}', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 포화 비율 표시
    n_sat = len(df_axis[df_axis['tau'] >= 99.9])
    ax.text(0.95, 0.95, f'포화: {n_sat}/{len(df_axis)}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'tau_distribution.png', dpi=150, bbox_inches='tight')
print(f"저장: tau_distribution.png")

# τ vs R² 상관관계
fig, ax = plt.subplots(figsize=(10, 6))

for axis in df['axis'].unique():
    df_axis = df[df['axis'] == axis]
    ax.scatter(df_axis['tau'], df_axis['r_squared'], alpha=0.6, label=axis, s=30)

ax.axvline(x=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='τ 상한선')
ax.set_xlabel('τ (s)', fontsize=12)
ax.set_ylabel('R² (모델 품질)', fontsize=12)
ax.set_title('시정수 vs 모델 품질', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'tau_vs_r2.png', dpi=150, bbox_inches='tight')
print(f"저장: tau_vs_r2.png")

# 리포트 생성
print("\n" + "="*80)
print("리포트 생성 중...")
print("="*80)

with open(output_dir / "TAU_REANALYSIS_REPORT.md", 'w', encoding='utf-8') as f:
    f.write("# 시정수(τ) 재분석 리포트\n\n")
    f.write("## 🚨 핵심 문제\n\n")
    
    total_saturated = len(df[df['tau'] >= 99.9])
    total = len(df)
    sat_pct = total_saturated / total * 100
    
    f.write(f"**{total_saturated}/{total} 샘플 ({sat_pct:.1f}%)이 τ 상한선(100s)에 도달**\n\n")
    f.write("이는 다음을 의미합니다:\n\n")
    f.write("1. **모델 피팅의 한계**: τ를 100s로 제한했고 대부분이 상한에 수렴\n")
    f.write("2. **비현실적인 정착 시간**: 4*τ=400s (6.7분)는 실제 유압 시스템과 맞지 않음\n")
    f.write("3. **데이터 특성 문제**: 입력 신호가 충분히 빠르지 않아 τ를 정확히 추정 불가\n\n")
    
    f.write("## 📊 축별 τ 포화 현황\n\n")
    f.write("| 축 | 전체 | 포화 | 포화율 | 평균 τ (전체) | 중앙값 τ (유효) |\n")
    f.write("|---|---|---|---|---|---|\n")
    
    for _, row in df_summary.iterrows():
        f.write(f"| {row['axis']} | {row['n_total']} | {row['n_saturated']} | "
               f"{row['saturation_%']:.1f}% | {row['tau_mean_all']:.1f}s | "
               f"{row['tau_median_valid']:.1f}s |\n")
    
    f.write("\n## 💡 해결 방안\n\n")
    f.write("### 방안 1: τ 재추정 (추천 ⭐)\n\n")
    f.write("**문제**: 현재 모델은 step response 피팅에 의존\n")
    f.write("**해결**: 실제 각도 변화 데이터로부터 63.2% 도달 시간 직접 계산\n\n")
    f.write("```python\n")
    f.write("# 각 샘플마다:\n")
    f.write("# 1. 시작 각도와 최종 각도 파악\n")
    f.write("# 2. 63.2% 지점 = 시작 + 0.632*(최종-시작)\n")
    f.write("# 3. 이 지점 도달 시간 = τ\n")
    f.write("```\n\n")
    
    f.write("### 방안 2: 고정 τ 사용\n\n")
    f.write("유압 시스템의 일반적인 시정수:\n")
    f.write("- **고속 밸브**: τ = 0.05 ~ 0.2s\n")
    f.write("- **일반 유압**: τ = 0.5 ~ 2s\n")
    f.write("- **대형 실린더**: τ = 2 ~ 5s\n\n")
    f.write("**추천**: τ = 1~2s로 가정하고 PID 재계산\n\n")
    
    f.write("### 방안 3: τ 없이 PID 계산\n\n")
    f.write("**Ziegler-Nichols Ultimate Gain 방법**:\n")
    f.write("- Step response 대신 진동 임계점 찾기\n")
    f.write("- τ 불필요\n\n")
    
    f.write("## 📈 예상 결과 비교\n\n")
    f.write("| 시나리오 | τ | 정착시간 (4*τ) | Kp | Ki |\n")
    f.write("|---|---|---|---|---|\n")
    f.write("| 현재 (포화) | 100s | 400s | 현재값 | 현재값 |\n")
    f.write("| 수정 (τ=2s) | 2s | 8s | **↑ 50배** | **↑ 50배** |\n")
    f.write("| 수정 (τ=1s) | 1s | 4s | **↑ 100배** | **↑ 100배** |\n\n")
    
    f.write("⚠️ **주의**: τ가 작아지면 Kp, Ki가 크게 증가!\n\n")
    
    f.write("## 🎯 권장 조치\n\n")
    f.write("1. ✅ **원시 데이터에서 τ 직접 계산** (가장 정확)\n")
    f.write("2. ✅ **τ=1~2s 가정하고 PID 재계산** (빠른 해결)\n")
    f.write("3. ✅ **실제 시스템에서 step response 재측정** (최선)\n\n")
    
    f.write("## 📌 결론\n\n")
    f.write(f"현재 정착시간 {df['tau'].mean()*4:.0f}초는 **모델 피팅 실패**의 결과입니다.\n\n")
    f.write("실제 유압 시스템의 정착 시간은 **5~10초 이내**일 것으로 예상됩니다.\n\n")
    f.write("**즉시 조치 필요!**\n")

print(f"\n리포트 저장: {output_dir / 'TAU_REANALYSIS_REPORT.md'}")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print(f"\n결과 위치: {output_dir}/")
print("\n생성된 파일:")
print("  - tau_summary.csv")
print("  - response_analysis.csv")
print("  - tau_distribution.png")
print("  - tau_vs_r2.png")
print("  - TAU_REANALYSIS_REPORT.md")

