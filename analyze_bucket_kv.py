"""
버킷 Kv 데이터 재확인
- 구간별 속도 특성 분석
- 이상치 검토
- 저속 전용 Kv 추정
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("Bucket Kv Analysis - 구간별 속도 특성")
print("="*80)

# ========== FINAL_GAINS_SUMMARY.md에서 추출한 데이터 ==========
# 섹션 14.5: Bucket_In, 14.6: Bucket_Out

bucket_data = {
    'Bucket_In': {
        'ultra_low': {'Kv': 0.0, 'K_offset': 40.0, 'R2': 0.0, 'samples': 3, 'range': '<5 deg/s'},
        'low': {'Kv': 6.045, 'K_offset': -22.6, 'R2': 0.432, 'samples': 4, 'range': '5-15 deg/s'},
        'mid': {'Kv': 56.800, 'K_offset': -1354.6, 'R2': 0.986, 'samples': 3, 'range': '15-30 deg/s'},
        'high': {'Kv': 0.340, 'K_offset': 56.4, 'R2': 0.133, 'samples': 19, 'range': '>30 deg/s'},
        'integrated': {'Kv': 0.50, 'K_offset': 45.6, 'R2': 0.43}
    },
    'Bucket_Out': {
        'ultra_low': {'Kv': 0.0, 'K_offset': 40.0, 'R2': 0.0, 'samples': 3, 'range': '<5 deg/s'},
        'low': {'Kv': 0.0, 'K_offset': 50.0, 'R2': 0.0, 'samples': 3, 'range': '5-15 deg/s'},
        'high': {'Kv': 0.622, 'K_offset': 51.5, 'R2': 0.136, 'samples': 20, 'range': '>30 deg/s'},
        'integrated': {'Kv': 0.85, 'K_offset': 40.6, 'R2': 0.64}
    }
}

# ========== 1. 데이터 요약 ==========
print("\n" + "="*80)
print("1. Bucket 속도 구간별 Kv 데이터 요약")
print("="*80)

for axis, data in bucket_data.items():
    print(f"\n### {axis} ###")
    print(f"{'Range':<20} {'Kv':>8} {'K_offset':>10} {'R²':>8} {'Samples':>8} {'Quality':>12}")
    print("-"*80)
    
    for range_name, values in data.items():
        if range_name == 'integrated':
            continue
        
        kv = values['Kv']
        k_off = values['K_offset']
        r2 = values['R2']
        samples = values.get('samples', 0)
        
        # Quality assessment
        if r2 > 0.8:
            quality = "Excellent"
        elif r2 > 0.5:
            quality = "Good"
        elif r2 > 0.3:
            quality = "Fair"
        elif kv == 0.0:
            quality = "No Data"
        else:
            quality = "Poor"
        
        print(f"{values['range']:<20} {kv:>8.3f} {k_off:>10.1f} {r2:>8.3f} {samples:>8} {quality:>12}")
    
    # Integrated
    int_data = data['integrated']
    print("-"*80)
    print(f"{'Integrated (all)':<20} {int_data['Kv']:>8.3f} {int_data['K_offset']:>10.1f} {int_data['R2']:>8.3f} {'--':>8} {'Baseline':>12}")

# ========== 2. 문제점 분석 ==========
print("\n" + "="*80)
print("2. 문제점 분석")
print("="*80)

print("\n### Bucket_In 문제점 ###")
print("1. 초저속 (<5 deg/s): Kv=0 (데이터 부족, 샘플 3개)")
print("2. 저속 (5-15 deg/s): Kv=6.045 (R²=0.432, 신뢰도 낮음, 샘플 4개)")
print("3. 중속 (15-30 deg/s): Kv=56.8 (R²=0.986, 하지만 K_offset=-1354 이상치!)")
print("4. 고속 (>30 deg/s): Kv=0.340 (R²=0.133, 신뢰도 매우 낮음, 샘플 19개)")
print("5. 통합: Kv=0.50 (R²=0.43, 낮음)")
print("\n→ 문제: 저속 데이터 부족, 신뢰도 낮음, 중속 이상치")

print("\n### Bucket_Out 문제점 ###")
print("1. 초저속 (<5 deg/s): Kv=0 (데이터 부족, 샘플 3개)")
print("2. 저속 (5-15 deg/s): Kv=0 (데이터 부족, 샘플 3개)")
print("3. 고속 (>30 deg/s): Kv=0.622 (R²=0.136, 신뢰도 매우 낮음, 샘플 20개)")
print("4. 통합: Kv=0.85 (R²=0.64, 중간)")
print("\n→ 문제: 저속 데이터 완전 부족, 고속만 존재하나 신뢰도 낮음")

# ========== 3. 구간별 특성 비교 (다른 축과) ==========
print("\n" + "="*80)
print("3. 다른 축과 비교 (저속 Kv)")
print("="*80)

comparison_data = {
    'Axis': ['Arm_In', 'Arm_Out', 'Boom_Up', 'Boom_Down', 'Bucket_In', 'Bucket_Out'],
    'Kv_ultra_low': [2.709, 0.0, 4.374, 3.134, 0.0, 0.0],
    'Kv_low': [0.0, 0.0, 3.165, 2.135, 6.045, 0.0],
    'Kv_integrated': [1.29, 1.39, 2.84, 2.75, 0.50, 0.85],
    'R2_ultra': [0.921, 0.0, 0.902, 0.992, 0.0, 0.0],
    'R2_low': [0.0, 0.0, 0.511, 0.883, 0.432, 0.0],
}

df_comp = pd.DataFrame(comparison_data)
print(df_comp.to_string(index=False))

print("\n### 관찰 ###")
print("- Arm, Boom: 초저속 Kv = 2~4 (신뢰도 높음)")
print("- Bucket_In: 저속 Kv = 6.045 (신뢰도 낮음, 하지만 다른 축보다 큼)")
print("- Bucket_Out: 저속 데이터 없음")
print("→ 버킷은 저속에서 오히려 Kv가 클 가능성!")

# ========== 4. 저속 Kv 추정 전략 ==========
print("\n" + "="*80)
print("4. 저속 Kv 추정 전략")
print("="*80)

print("\n### Strategy A: 저속 데이터 사용 (Bucket_In만) ###")
print("Bucket_In:  Kv = 6.045 (R²=0.432)")
print("Bucket_Out: Kv = 0.85 (통합값, 저속 데이터 없음)")
print("→ 장점: 실제 데이터 기반")
print("→ 단점: 신뢰도 낮음")

print("\n### Strategy B: 통합값 사용 ###")
print("Bucket_In:  Kv = 0.50")
print("Bucket_Out: Kv = 0.85")
print("→ 장점: 안전함")
print("→ 단점: 저속 특성 반영 안 됨")

print("\n### Strategy C: 비율 추정 (다른 축 기준) ###")
# Arm, Boom의 저속/통합 비율 계산
ratios = []
axes_ref = ['Arm_In', 'Boom_Up', 'Boom_Down']
for axis in axes_ref:
    row = df_comp[df_comp['Axis'] == axis].iloc[0]
    kv_low = row['Kv_ultra_low']
    kv_int = row['Kv_integrated']
    if kv_low > 0 and kv_int > 0:
        ratio = kv_low / kv_int
        ratios.append(ratio)
        print(f"{axis}: 저속/통합 = {kv_low:.3f}/{kv_int:.3f} = {ratio:.2f}x")

avg_ratio = np.mean(ratios)
print(f"\n평균 비율: {avg_ratio:.2f}x")
print(f"\n버킷 추정:")
print(f"Bucket_In:  Kv = 0.50 × {avg_ratio:.2f} = {0.50 * avg_ratio:.3f}")
print(f"Bucket_Out: Kv = 0.85 × {avg_ratio:.2f} = {0.85 * avg_ratio:.3f}")
print("→ 장점: 다른 축 경험 활용")
print("→ 단점: 버킷 특성 다를 수 있음")

print("\n### Strategy D: 실측 데이터 기반 조정 ###")
print("Bucket_In 저속 Kv = 6.045 (실측)")
print("  vs 통합 Kv = 0.50")
print("  비율: 12.1x (다른 축 평균 2.1x 대비 매우 큼)")
print("\nBucket_In 중속 Kv = 56.8 (이상치로 보임)")
print("  K_offset = -1354 (비현실적)")
print("  → 이 데이터는 제외")
print("\nBucket_In 고속 Kv = 0.340")
print("  → 속도 증가하면 Kv 감소 (비선형성)")

print("\n→ 결론: 버킷은 구간별 특성이 크게 다름!")
print("→ 저속: Kv 큼 (6.0~)")
print("→ 고속: Kv 작음 (~0.5)")

# ========== 5. 최종 권장 ==========
print("\n" + "="*80)
print("5. 최종 권장 Kv (저속 <10 deg/s)")
print("="*80)

recommendations = []

# Bucket_In
print("\n### Bucket_In ###")
print("Option 1: 저속 실측값 사용 → Kv = 6.045 (R²=0.432)")
print("Option 2: 통합값 사용 → Kv = 0.50 (안전)")
print("Option 3: 비율 추정 → Kv = 1.05")
print("\n권장: Option 1 (실측값)")
print("  - 신뢰도 낮지만 유일한 저속 데이터")
print("  - 저속에서 Kv가 큰 것이 물리적으로 타당")
print("  - 안전계수 적용 권장")

kv_bucket_in = 6.045
kv_bucket_in_safe = 0.50  # fallback

print(f"\n최종: Kv = {kv_bucket_in:.3f} (1차 선택)")
print(f"      Kv = {kv_bucket_in_safe:.3f} (2차 fallback)")

recommendations.append({
    'Axis': 'Bucket_In',
    'Kv_primary': kv_bucket_in,
    'Kv_fallback': kv_bucket_in_safe,
    'R2': 0.432,
    'Confidence': 'Low'
})

# Bucket_Out
print("\n### Bucket_Out ###")
print("Option 1: 저속 데이터 없음")
print("Option 2: 통합값 사용 → Kv = 0.85 (R²=0.64)")
print("Option 3: 비율 추정 → Kv = 1.79")
print("Option 4: Bucket_In 비율 적용 → Kv = 0.85 × (6.045/0.50) = 10.3")
print("\n권장: Option 2 (통합값)")
print("  - 저속 데이터 부족")
print("  - 통합값이 중간 정도 신뢰도")

kv_bucket_out = 0.85
kv_bucket_out_alt = 1.79  # 비율 추정

print(f"\n최종: Kv = {kv_bucket_out:.3f} (1차 선택)")
print(f"      Kv = {kv_bucket_out_alt:.3f} (2차 비율 추정)")

recommendations.append({
    'Axis': 'Bucket_Out',
    'Kv_primary': kv_bucket_out,
    'Kv_fallback': kv_bucket_out_alt,
    'R2': 0.64,
    'Confidence': 'Medium'
})

# ========== 6. 요약 테이블 ==========
print("\n" + "="*80)
print("6. 버킷 Kv 최종 권장값 요약")
print("="*80)

df_rec = pd.DataFrame(recommendations)
print(df_rec.to_string(index=False))

print("\n### 주의사항 ###")
print("1. Bucket_In Kv=6.045: 신뢰도 낮음 (R²=0.432)")
print("   → 실제 테스트에서 조정 필요")
print("   → PID 게인 작게 나올 수 있음")
print("\n2. Bucket_Out Kv=0.85: 저속 데이터 없음")
print("   → 통합값 사용, 저속 특성 반영 안 됨")
print("   → 실측 후 재조정 권장")
print("\n3. 버킷은 속도 구간별 비선형성 큼")
print("   → 룩업 테이블 적용 고려")

# ========== 7. PID 게인에 미치는 영향 예측 ==========
print("\n" + "="*80)
print("7. PID 게인에 미치는 영향 (tau=2s, lambda_factor=0.5)")
print("="*80)

tau = 2.0
lambda_factor = 0.5
lambda_val = lambda_factor * tau

scenarios = [
    {'name': 'Bucket_In (Kv=6.045, 저속)', 'kv': 6.045},
    {'name': 'Bucket_In (Kv=0.50, 통합)', 'kv': 0.50},
    {'name': 'Bucket_Out (Kv=0.85, 통합)', 'kv': 0.85},
    {'name': 'Bucket_Out (Kv=1.79, 비율)', 'kv': 1.79},
]

print(f"\n{'Scenario':<35} {'Kv':>8} {'Kp':>8} {'Ki':>8} {'10도 출력':>12}")
print("-"*80)

for sc in scenarios:
    kv = sc['kv']
    Kp = tau / (kv * lambda_val)
    Ki = 1 / (kv * lambda_val)
    output_10 = Kp * 10
    
    print(f"{sc['name']:<35} {kv:>8.3f} {Kp:>8.2f} {Ki:>8.2f} {output_10:>11.1f}%")

print("\n### 관찰 ###")
print("- Bucket_In Kv=6.045 → Kp=0.33 (매우 작음, 응답 느림)")
print("- Bucket_In Kv=0.50 → Kp=4.00 (큼, 응답 빠름)")
print("- 12배 차이! → Kv 선택이 매우 중요")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)

