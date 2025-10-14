"""
속도 기반 Kv를 사용한 PID 게인 재계산
저속 동작(<10 deg/s) 전용
"""

import numpy as np
import pandas as pd

# ========== 1. 저속 Kv 데이터 ==========
kv_data = {
    'Axis': ['Arm_In', 'Arm_Out', 'Boom_Up', 'Boom_Down', 'Bucket_In', 'Bucket_Out'],
    'Kv_ultra_low': [2.709, np.nan, 4.374, 3.134, np.nan, np.nan],  # <5 deg/s
    'R2_ultra': [0.921, 0.0, 0.902, 0.992, 0.0, 0.0],
    'Kv_integrated': [1.29, 1.39, 2.84, 2.75, 0.50, 0.85],  # All speeds
}

df = pd.DataFrame(kv_data)

# 초저속 Kv 우선, 없으면 통합값 사용
df['Kv_selected'] = df['Kv_ultra_low'].fillna(df['Kv_integrated'])

print("="*80)
print("1. Selected Kv for Low Speed (<10 deg/s)")
print("="*80)
print(df[['Axis', 'Kv_selected', 'R2_ultra']].to_string(index=False))

# ========== 2. PID 게인 계산 (여러 lambda_factor 시도) ==========
tau = 2.0  # 시정수 (s)
lambda_factors = [0.2, 0.3, 0.5, 1.0, 2.0]

print("\n" + "="*80)
print("2. PID Gains with Different lambda_factors")
print("="*80)

results_all = []
for lf in lambda_factors:
    lambda_val = lf * tau
    
    results_lf = []
    for _, row in df.iterrows():
        axis = row['Axis']
        kv = row['Kv_selected']
        
        Kp = tau / (kv * lambda_val)
        Ki = 1 / (kv * lambda_val)
        Kd = 0.0
        
        results_lf.append({
            'Axis': axis,
            'Kv': kv,
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
        })
        
        results_all.append({
            'lambda_factor': lf,
            'Axis': axis,
            'Kv': kv,
            'Kp': Kp,
            'Ki': Ki,
        })
    
    print(f"\n--- lambda_factor = {lf} (lambda = {lambda_val}s) ---")
    df_lf = pd.DataFrame(results_lf)
    print(df_lf.to_string(index=False))

# ========== 3. V3 실제 결과와 비교 ==========
v3_actual = {
    'Arm_In': {'Kp': 3.74, 'Ki': 1.94},
    'Arm_Out': {'Kp': 3.20, 'Ki': 1.60},
    'Boom_Up': {'Kp': 11.16, 'Ki': 5.58},
    'Boom_Down': {'Kp': 5.90, 'Ki': 2.95},
    'Bucket_In': {'Kp': 1.86, 'Ki': 0.93},
    'Bucket_Out': {'Kp': 1.55, 'Ki': 0.78}
}

print("\n" + "="*80)
print("3. Comparison with V3 Actual (tau=2s)")
print("="*80)

for lf in [0.2, 0.5]:
    print(f"\n--- lambda_factor = {lf} ---")
    print(f"{'Axis':<12} {'V3 Kp':>8} {'New Kp':>8} {'Ratio':>8} | {'V3 Ki':>8} {'New Ki':>8} {'Ratio':>8}")
    print("-"*75)
    
    df_lf = pd.DataFrame(results_all)
    df_lf = df_lf[df_lf['lambda_factor'] == lf]
    
    for _, row in df_lf.iterrows():
        axis = row['Axis']
        v3 = v3_actual[axis]
        
        kp_ratio = row['Kp'] / v3['Kp']
        ki_ratio = row['Ki'] / v3['Ki']
        
        print(f"{axis:<12} {v3['Kp']:>8.2f} {row['Kp']:>8.2f} {kp_ratio:>7.2f}x | "
              f"{v3['Ki']:>8.2f} {row['Ki']:>8.2f} {ki_ratio:>7.2f}x")

# ========== 4. V3 역산: 어떤 lambda_factor를 사용했을까? ==========
print("\n" + "="*80)
print("4. Reverse Engineering: What lambda_factor did V3 use?")
print("="*80)

print(f"{'Axis':<12} {'Kv':>8} {'V3 Kp':>8} {'V3 Ki':>8} | {'Implied λ_factor':>10} {'Implied λ (s)':>10}")
print("-"*80)

for _, row in df.iterrows():
    axis = row['Axis']
    kv = row['Kv_selected']
    v3 = v3_actual[axis]
    
    # Kp = tau / (Kv * lambda) => lambda = tau / (Kv * Kp)
    lambda_implied = tau / (kv * v3['Kp'])
    lambda_factor_implied = lambda_implied / tau
    
    print(f"{axis:<12} {kv:>8.3f} {v3['Kp']:>8.2f} {v3['Ki']:>8.2f} | "
          f"{lambda_factor_implied:>10.3f} {lambda_implied:>10.2f}")

# ========== 5. 권장 lambda_factor 결정 ==========
print("\n" + "="*80)
print("5. Recommendation")
print("="*80)

# V3 역산 평균
lambda_factors_implied = []
for _, row in df.iterrows():
    axis = row['Axis']
    kv = row['Kv_selected']
    v3 = v3_actual[axis]
    lambda_implied = tau / (kv * v3['Kp'])
    lambda_factor_implied = lambda_implied / tau
    lambda_factors_implied.append(lambda_factor_implied)

lambda_factor_avg = np.mean(lambda_factors_implied)
lambda_factor_median = np.median(lambda_factors_implied)

print(f"V3 implied lambda_factor (average): {lambda_factor_avg:.3f}")
print(f"V3 implied lambda_factor (median):  {lambda_factor_median:.3f}")
print(f"\nRecommendation: Use lambda_factor = {lambda_factor_median:.2f}")
print(f"(This will give PID gains closest to V3 actual values)")

# ========== 6. 최종 권장 게인 계산 ==========
lambda_factor_final = round(lambda_factor_median, 2)
lambda_val_final = lambda_factor_final * tau

print("\n" + "="*80)
print(f"6. Final PID Gains (lambda_factor = {lambda_factor_final})")
print("="*80)

final_gains = []
for _, row in df.iterrows():
    axis = row['Axis']
    kv = row['Kv_selected']
    
    Kp = tau / (kv * lambda_val_final)
    Ki = 1 / (kv * lambda_val_final)
    Kd = 0.0
    
    # 10도 오차 시 출력
    output_10deg = Kp * 10
    
    # 1초 후 I 출력
    output_1s_i = Ki * 10 * 1.0
    
    final_gains.append({
        'Axis': axis,
        'Direction': axis.split('_')[1],
        'Kv': kv,
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd,
        'settling_time': 4 * tau,
        'output_10deg': output_10deg,
        'output_1s_I': output_1s_i
    })

df_final = pd.DataFrame(final_gains)
print(df_final.to_string(index=False))

# ========== 7. 안전 게인 (50% Ki) ==========
print("\n" + "="*80)
print("7. Safe Gains (50% Ki for initial testing)")
print("="*80)

safe_gains = []
for _, row in df_final.iterrows():
    safe_gains.append({
        'Axis': row['Axis'],
        'Kp': row['Kp'],
        'Ki': row['Ki'] * 0.5,
        'Kd': 0.0
    })

df_safe = pd.DataFrame(safe_gains)
print(df_safe.to_string(index=False))

print("\n" + "="*80)
print("Calculation Complete!")
print("="*80)

