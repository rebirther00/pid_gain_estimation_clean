"""
각도 기반 룩업 테이블 필요성 검토
이전 분석 결과: 모든 duty 레벨이 같은 각도 범위를 커버
→ Kv 변화는 각도가 아닌 속도(duty)에 의한 것
"""

import pandas as pd
from pathlib import Path

base = Path(__file__).parent.parent
output_dir = base / "output" / "angle_dependency_check"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("각도 기반 vs 속도 기반 룩업 테이블 검토")
print("="*80)

# Load data
df = pd.read_csv(base / "output" / "ff_lookup_detailed" / "ff_by_velocity_range.csv")

print("\n=== 핵심 발견 (이전 분석) ===\n")
print("1. 모든 duty 레벨(40-100%)이 같은 각도 범위를 커버")
print("2. Kv 변화는 '각도'가 아닌 '속도'에 의한 것")
print("3. Duty = 속도의 간접 지표 (높은 duty → 높은 속도)")

print("\n=== 속도 범위별 FF 성능 (다시 확인) ===\n")

for axis in df['axis'].unique():
    df_axis = df[df['axis'] == axis]
    
    print(f"\n{axis}:")
    print(f"  속도 구간 수: {len(df_axis)}")
    
    kv_min = df_axis['Kv'].min()
    kv_max = df_axis['Kv'].max()
    
    if kv_min != 0:
        kv_variation = (kv_max - kv_min) / abs(kv_min) * 100
    else:
        kv_variation = float('inf')
    
    print(f"  Kv 범위: {kv_min:.3f} ~ {kv_max:.3f}")
    print(f"  Kv 변동률: {kv_variation:.1f}%")
    
    for _, row in df_axis.iterrows():
        print(f"    {row['velocity_range']:20s}: Kv={row['Kv']:6.3f}, R²={row['R2_ff']:.3f}")

print("\n" + "="*80)
print("결론")
print("="*80)

print("""
✅ 각도 기반 룩업 테이블: 불필요
   - 이유: 모든 각도 범위에서 duty 40~100% 테스트 수행
   - 각도 변화 없이도 속도가 변함 (duty로 제어)

✅ 속도 기반 룩업 테이블: 필수
   - 이유: 속도(=duty)에 따라 Kv가 크게 변함
   - 저속: 높은 Kv (세밀한 제어)
   - 고속: 낮은 Kv (빠른 이동)

📌 핵심 이해:
   Duty → 속도 → Kv 변화
   (각도는 Kv에 직접 영향 없음)
""")

print("\n파일 저장: output/angle_dependency_check/")

with open(output_dir / "ANGLE_VS_VELOCITY_CONCLUSION.md", 'w', encoding='utf-8') as f:
    f.write("# 각도 기반 vs 속도 기반 룩업 테이블\n\n")
    f.write("## 핵심 질문\n\n")
    f.write("**FF 게인에 대해서 각도 범위 기반 룩업 테이블은 검토가 필요 없는지?**\n\n")
    f.write("## 답변: 불필요 ✅\n\n")
    
    f.write("### 이유\n\n")
    f.write("1. **모든 duty 레벨(40-100%)이 동일한 각도 범위를 커버**\n")
    f.write("   - 예: Arm In은 duty 40~100% 모두 '40도 → -80도' 이동\n")
    f.write("   - 각도 범위는 고정, duty만 변화\n\n")
    
    f.write("2. **Kv 변화는 속도에 의한 것**\n")
    f.write("   - 낮은 duty (40%) → 느린 속도 → 높은 Kv\n")
    f.write("   - 높은 duty (100%) → 빠른 속도 → 낮은 Kv\n")
    f.write("   - 각도는 직접적인 영향 없음\n\n")
    
    f.write("3. **Duty는 속도의 간접 지표**\n")
    f.write("   - Duty ↑ → 유량 ↑ → 속도 ↑\n")
    f.write("   - 실제 제어는 목표 속도를 입력받음\n\n")
    
    f.write("### 실험적 증거\n\n")
    f.write("| 축 | 속도 범위별 Kv 변동 | 각도 범위 | 결론 |\n")
    f.write("|---|---|---|---|\n")
    
    for axis in df['axis'].unique():
        df_axis = df[df['axis'] == axis]
        kv_min = df_axis['Kv'].min()
        kv_max = df_axis['Kv'].max()
        
        f.write(f"| {axis} | {kv_min:.3f} ~ {kv_max:.3f} | 동일 | 속도 의존 |\n")
    
    f.write("\n### 최종 결론\n\n")
    f.write("✅ **각도 기반 룩업 불필요**\n")
    f.write("✅ **속도 기반 룩업 필수** (3단계: <5, 5-15, >15 deg/s)\n\n")
    
    f.write("### 구현 예시\n\n")
    f.write("```cpp\n")
    f.write("// 올바른 방법: 속도 기반\n")
    f.write("double getKv(double velocity) {\n")
    f.write("    if (abs(velocity) < 5.0)\n")
    f.write("        return 3.13;  // 저속\n")
    f.write("    else if (abs(velocity) < 15.0)\n")
    f.write("        return 2.13;  // 중속\n")
    f.write("    else\n")
    f.write("        return 1.33;  // 고속\n")
    f.write("}\n\n")
    f.write("// 잘못된 방법: 각도 기반\n")
    f.write("// double getKv(double angle) { ... }  // ❌ 불필요!\n")
    f.write("```\n")

print("저장 완료: ANGLE_VS_VELOCITY_CONCLUSION.md")

