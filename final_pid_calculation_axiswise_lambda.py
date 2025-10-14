"""
최종 PID 게인 계산 - 축별 lambda_factor 적용
저속 동작(<10 deg/s) 전용, 속도 기반 Kv 사용
"""

import numpy as np
import pandas as pd

print("="*80)
print("최종 PID 게인 계산 - 축별 lambda_factor")
print("="*80)

# ========== 1. 저속 Kv 데이터 (버킷 재검토 반영) ==========
kv_data = {
    'Arm_In': {'Kv': 2.709, 'R2': 0.921, 'source': 'ultra_low'},
    'Arm_Out': {'Kv': 1.390, 'R2': 0.0, 'source': 'integrated'},
    'Boom_Up': {'Kv': 4.374, 'R2': 0.902, 'source': 'ultra_low'},
    'Boom_Down': {'Kv': 3.134, 'R2': 0.992, 'source': 'ultra_low'},
    'Bucket_In': {'Kv': 6.045, 'R2': 0.432, 'source': 'low'},  # 저속 실측
    'Bucket_Out': {'Kv': 0.850, 'R2': 0.640, 'source': 'integrated'}  # 통합
}

print("\n1. 저속 Kv 선택 (버킷 재검토 완료)")
print("-"*80)
print(f"{'Axis':<12} {'Kv':>8} {'R²':>8} {'Source':>15} {'Quality':>12}")
print("-"*80)
for axis, data in kv_data.items():
    kv = data['Kv']
    r2 = data['R2']
    source = data['source']
    
    if r2 > 0.8:
        quality = "Excellent"
    elif r2 > 0.5:
        quality = "Good"
    elif r2 > 0.3:
        quality = "Fair"
    else:
        quality = "Fallback"
    
    print(f"{axis:<12} {kv:>8.3f} {r2:>8.3f} {source:>15} {quality:>12}")

# ========== 2. V3 실제 게인 (tau=2s 기준) ==========
v3_actual = {
    'Arm_In': {'Kp': 3.74, 'Ki': 1.94},
    'Arm_Out': {'Kp': 3.20, 'Ki': 1.60},
    'Boom_Up': {'Kp': 11.16, 'Ki': 5.58},
    'Boom_Down': {'Kp': 5.90, 'Ki': 2.95},
    'Bucket_In': {'Kp': 1.86, 'Ki': 0.93},
    'Bucket_Out': {'Kp': 1.55, 'Ki': 0.78}
}

# ========== 3. 축별 lambda_factor 역산 ==========
tau = 2.0

print("\n2. V3 역산: 축별 lambda_factor")
print("-"*80)
print(f"{'Axis':<12} {'Kv':>8} {'V3 Kp':>8} {'V3 Ki':>8} | {'λ_factor':>10} {'λ (s)':>8}")
print("-"*80)

lambda_factors_implied = {}
for axis, kv_info in kv_data.items():
    kv = kv_info['Kv']
    v3 = v3_actual[axis]
    
    # Kp = tau / (Kv * lambda) => lambda = tau / (Kv * Kp)
    lambda_implied = tau / (kv * v3['Kp'])
    lambda_factor_implied = lambda_implied / tau
    
    lambda_factors_implied[axis] = lambda_factor_implied
    
    print(f"{axis:<12} {kv:>8.3f} {v3['Kp']:>8.2f} {v3['Ki']:>8.2f} | {lambda_factor_implied:>10.3f} {lambda_implied:>8.3f}")

# 통계
all_lambdas = list(lambda_factors_implied.values())
print("-"*80)
print(f"{'통계':<12} {'평균':>8} {'중앙값':>8} {'최소':>8} {'최대':>8}")
print(f"{'lambda_factor':<12} {np.mean(all_lambdas):>8.3f} {np.median(all_lambdas):>8.3f} "
      f"{np.min(all_lambdas):>8.3f} {np.max(all_lambdas):>8.3f}")

# ========== 4. 축별 lambda_factor 조정 전략 ==========
print("\n3. 축별 lambda_factor 조정 전략")
print("-"*80)

# 역산값을 기준으로 조정
lambda_factors_final = {}

for axis, lambda_implied in lambda_factors_implied.items():
    # Boom_Up은 특별 처리 (중력 반대 방향, 높은 Kp 필요)
    if axis == 'Boom_Up':
        lambda_adjusted = 0.02  # V3와 일치하도록 (Kp=11.16)
    else:
        # 너무 작은 값은 0.05로 하한선
        # 너무 큰 값은 1.0으로 상한선
        lambda_adjusted = np.clip(lambda_implied, 0.05, 1.0)
        
        # 반올림 (0.05 단위)
        lambda_adjusted = round(lambda_adjusted / 0.05) * 0.05
    
    lambda_factors_final[axis] = lambda_adjusted
    
    adjustment = ""
    if axis == 'Boom_Up':
        adjustment = f" (원래 {lambda_implied:.3f} → 0.02 특별 조정, 중력 보상)"
    elif lambda_adjusted != lambda_implied:
        if lambda_implied < 0.05:
            adjustment = f" (원래 {lambda_implied:.3f} → 0.05 하한선)"
        elif lambda_implied > 1.0:
            adjustment = f" (원래 {lambda_implied:.3f} → 1.0 상한선)"
    
    print(f"{axis:<12} λ_factor = {lambda_adjusted:.2f}{adjustment}")

# ========== 5. 최종 PID 게인 계산 ==========
print("\n4. 최종 PID 게인 (축별 lambda_factor)")
print("="*80)

final_gains = []

for axis, kv_info in kv_data.items():
    kv = kv_info['Kv']
    lambda_factor = lambda_factors_final[axis]
    lambda_val = lambda_factor * tau
    
    # PID 계산
    Kp = tau / (kv * lambda_val)
    Ki = 1 / (kv * lambda_val)
    Kd = 0.0
    
    # 성능 지표
    settling_time = 4 * tau  # 4τ
    output_10deg = Kp * 10  # 10도 오차 시 출력
    output_1s_I = Ki * 10 * 1.0  # 1초 후 I 출력
    
    final_gains.append({
        'Axis': axis,
        'Direction': axis.split('_')[1],
        'Kv': kv,
        'lambda_factor': lambda_factor,
        'lambda': lambda_val,
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd,
        'Ki_Kp_ratio': Ki / Kp,
        'settling_time_s': settling_time,
        'output_10deg_%': output_10deg,
        'output_1s_I_%': output_1s_I
    })

df_final = pd.DataFrame(final_gains)

print("\n### 전체 게인 ###")
print(df_final[['Axis', 'Kv', 'lambda_factor', 'Kp', 'Ki', 'Kd']].to_string(index=False))

print("\n### 성능 지표 ###")
print(df_final[['Axis', 'settling_time_s', 'output_10deg_%', 'output_1s_I_%']].to_string(index=False))

print("\n### Ki/Kp 비율 확인 ###")
print(df_final[['Axis', 'Ki_Kp_ratio', 'lambda']].to_string(index=False))
print(f"\n이론값: Ki/Kp = 1/τ = 1/{tau} = {1/tau:.2f}")

# ========== 6. V3 실제값과 비교 ==========
print("\n5. V3 실제값과 비교")
print("="*80)
print(f"{'Axis':<12} {'Kv':>8} {'λ_f':>6} | {'V3 Kp':>8} {'New Kp':>8} {'Ratio':>7} | "
      f"{'V3 Ki':>8} {'New Ki':>8} {'Ratio':>7}")
print("-"*80)

comparison = []
for _, row in df_final.iterrows():
    axis = row['Axis']
    v3 = v3_actual[axis]
    
    kp_ratio = row['Kp'] / v3['Kp']
    ki_ratio = row['Ki'] / v3['Ki']
    
    print(f"{axis:<12} {row['Kv']:>8.3f} {row['lambda_factor']:>6.2f} | "
          f"{v3['Kp']:>8.2f} {row['Kp']:>8.2f} {kp_ratio:>6.2f}x | "
          f"{v3['Ki']:>8.2f} {row['Ki']:>8.2f} {ki_ratio:>6.2f}x")
    
    comparison.append({
        'Axis': axis,
        'Kp_ratio': kp_ratio,
        'Ki_ratio': ki_ratio,
        'Kp_diff_%': (kp_ratio - 1) * 100,
        'Ki_diff_%': (ki_ratio - 1) * 100
    })

df_comp = pd.DataFrame(comparison)

print("\n### 통계 ###")
print(f"Kp 평균 비율: {df_comp['Kp_ratio'].mean():.2f}x (평균 {df_comp['Kp_diff_%'].mean():+.1f}%)")
print(f"Ki 평균 비율: {df_comp['Ki_ratio'].mean():.2f}x (평균 {df_comp['Ki_diff_%'].mean():+.1f}%)")

# ========== 7. 안전 게인 (50% Ki) ==========
print("\n6. 안전 게인 (50% Ki, 초기 테스트용)")
print("="*80)

safe_gains = []
for _, row in df_final.iterrows():
    safe_gains.append({
        'Axis': row['Axis'],
        'Direction': row['Direction'],
        'Kp': row['Kp'],
        'Ki': row['Ki'] * 0.5,  # 50%
        'Kd': 0.0,
        'Note': '초기 테스트용'
    })

df_safe = pd.DataFrame(safe_gains)
print(df_safe.to_string(index=False))

# ========== 8. YAML 형식 출력 ==========
print("\n7. 제어기 구현용 (YAML)")
print("="*80)

print("\n# 최종 PID 게인 (속도 기반, 축별 lambda_factor)")
print("PID_Gains_Final:")
for _, row in df_final.iterrows():
    axis_name = row['Axis'].replace('_', '_')
    print(f"  {axis_name}:")
    print(f"    Kp: {row['Kp']:.3f}  # %/deg")
    print(f"    Ki: {row['Ki']:.3f}  # %/(deg*s)")
    print(f"    Kd: {row['Kd']:.3f}")
    print(f"    # Kv: {row['Kv']:.3f}, lambda_factor: {row['lambda_factor']:.2f}")

print("\nPID_Gains_Safe:")
for _, row in df_safe.iterrows():
    axis_name = row['Axis'].replace('_', '_')
    print(f"  {axis_name}:")
    print(f"    Kp: {row['Kp']:.3f}")
    print(f"    Ki: {row['Ki']:.3f}  # 50% of final")
    print(f"    Kd: {row['Kd']:.3f}")

# ========== 9. FF 게인 ==========
print("\n8. FF 게인 (저속 전용)")
print("="*80)

ff_gains = {
    'Arm_In': {'Kv': 2.709, 'K_offset': 36.0},
    'Arm_Out': {'Kv': 1.390, 'K_offset': 40.2},
    'Boom_Up': {'Kv': 4.374, 'K_offset': 35.9},
    'Boom_Down': {'Kv': 3.134, 'K_offset': 35.4},
    'Bucket_In': {'Kv': 6.045, 'K_offset': -22.6},  # K_offset 주의!
    'Bucket_Out': {'Kv': 0.850, 'K_offset': 40.6}
}

print("\nFF_Gains_LowSpeed:")
for axis, ff in ff_gains.items():
    axis_name = axis.replace('_', '_')
    print(f"  {axis_name}:")
    print(f"    Kv: {ff['Kv']:.3f}  # (%/deg/s)")
    print(f"    K_offset: {ff['K_offset']:.1f}  # %")

# ========== 10. 주의사항 ==========
print("\n9. 주의사항")
print("="*80)
print("1. Bucket_In:")
print("   - Kv=6.045 (R²=0.432, 신뢰도 낮음)")
print("   - K_offset=-22.6 (음수 주의!)")
print("   - 실측 후 조정 필수")
print("\n2. Bucket_Out:")
print("   - Kv=0.85 (통합값, 저속 데이터 없음)")
print("   - 저속 특성 미반영")
print("\n3. Boom_Up:")
print("   - lambda_factor=0.02 (매우 작음, 매우 빠른 응답)")
print("   - Kp=11.4 (높음, 중력 반대 방향 보상)")
print("   - 10도 오차 → 114% duty (포화 주의)")
print("   - Kv=4.374 (Down보다 큼, 유압 특성 or 카운터밸런스)")
print("\n4. 모든 축:")
print("   - Anti-windup 필수 (integral ∈ [-10, 10])")
print("   - Duty saturation (∈ [0, 100])")
print("   - Ki는 50%부터 시작 권장")

print("\n" + "="*80)
print("계산 완료!")
print("="*80)

# CSV 저장
df_final.to_csv('final_pid_gains_axiswise.csv', index=False, encoding='utf-8-sig')
df_safe.to_csv('safe_pid_gains_axiswise.csv', index=False, encoding='utf-8-sig')
print("\n파일 저장:")
print("  - final_pid_gains_axiswise.csv")
print("  - safe_pid_gains_axiswise.csv")

