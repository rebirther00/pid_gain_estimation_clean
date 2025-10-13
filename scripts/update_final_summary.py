"""
FINAL_GAINS_SUMMARY.md에 τ=2s 게인 및 속도별 FF 룩업 테이블 추가
"""

import os
os.system('chcp 65001 >nul')

import json
import pandas as pd
from pathlib import Path

base = Path(__file__).parent.parent

print("="*80)
print("FINAL_GAINS_SUMMARY.md 업데이트")
print("="*80)

# Load τ=2s gains
tau2_file = base / "output" / "pid_tau2" / "final_gains_tau2.json"
if not tau2_file.exists():
    print(f"오류: {tau2_file} 파일이 없습니다!")
    exit(1)

with open(tau2_file, 'r') as f:
    tau2_gains = json.load(f)

# Load FF lookup data
ff_lookup_file = base / "output" / "ff_lookup_detailed" / "ff_by_velocity_range.csv"
if not ff_lookup_file.exists():
    print(f"경고: {ff_lookup_file} 파일이 없습니다. FF 룩업 테이블 스킵.")
    ff_lookup_df = None
else:
    ff_lookup_df = pd.read_csv(ff_lookup_file)

# Read current FINAL_GAINS_SUMMARY.md
summary_file = base / "FINAL_GAINS_SUMMARY.md"
if not summary_file.exists():
    print(f"오류: {summary_file} 파일이 없습니다!")
    exit(1)

with open(summary_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if sections already exist
if "## 📋 13. τ=2s 기준 최종 PID 게인" in content:
    print("섹션 13, 14가 이미 존재합니다. 건너뜀.")
    exit(0)

# Find insertion point (before the final line)
marker = "**완료 날짜**: 2025-10-13 (추가 분석 포함) 🎉"
if marker not in content:
    # Try alternative marker
    marker = "**생성 파일**:"
    if marker not in content:
        print("오류: 삽입 위치를 찾을 수 없습니다!")
        exit(1)

# Create new section for τ=2s gains
new_section = "\n\n---\n\n"
new_section += "## 📋 13. τ=2s 기준 최종 PID 게인 (실제 적용 권장)\n\n"
new_section += "### 13.1 개선 사항\n\n"
new_section += "기존 τ=100s는 상한선 포화로 **비현실적**이므로, **τ=2s로 재계산**했습니다.\n\n"
new_section += "| 항목 | 기존 (τ=100s) | 수정 (τ=2s) | 개선율 |\n"
new_section += "|---|---|---|---|\n"
new_section += "| **시정수** | 100s | **2s** | **98%** ↓ |\n"
new_section += "| **정착시간** | 400s (6.7분) | **8s** | **98%** ↓ |\n"
new_section += "| **Kp 변화** | - | **0%** (유지) | ✅ |\n"
new_section += "| **Ki 변화** | - | **+4900%** (50배) | ✅ |\n"
new_section += "| **Ki/Kp 비율** | 0.01 | **0.5** | 1/τ |\n\n"

new_section += "### 13.2 최종 PID 게인 (τ=2s)\n\n"
new_section += "**즉시 적용 가능한 게인입니다!**\n\n"
new_section += "| 축 | 방향 | Kp | Ki | Kd | 정착시간 | 10도 오차 출력 |\n"
new_section += "|---|------|-------|--------|-------|---------|----------------|\n"

axis_order = [
    ("Arm", "In"), ("Arm", "Out"),
    ("Boom", "Up"), ("Boom", "Down"),
    ("Bucket", "In"), ("Bucket", "Out")
]

for axis_name, direction in axis_order:
    axis_key = f"{axis_name}_{direction}"
    if axis_key in tau2_gains:
        gains = tau2_gains[axis_key]
        Kp = gains['Kp_new']
        Ki = gains['Ki_new']
        settling = gains['settling_time_new']
        output_10deg = Kp * 10
        
        new_section += f"| **{axis_name}** | {direction} | **{Kp:.2f}** | **{Ki:.2f}** | 0.0 | {settling:.0f}s | {output_10deg:.1f}% |\n"

new_section += "\n### 13.3 안전 게인 (50% Ki)\n\n"
new_section += "**첫 테스트 시 권장하는 안전 게인입니다.**\n\n"
new_section += "| 축 | 방향 | Kp | Ki (50%) | Kd |\n"
new_section += "|---|------|-------|----------|-----|\n"

for axis_name, direction in axis_order:
    axis_key = f"{axis_name}_{direction}"
    if axis_key in tau2_gains:
        gains = tau2_gains[axis_key]
        Kp = gains['Kp_new']
        Ki = gains['Ki_new'] * 0.5
        
        new_section += f"| **{axis_name}** | {direction} | **{Kp:.2f}** | **{Ki:.2f}** | 0.0 |\n"

new_section += "\n### 13.4 핵심 발견\n\n"
new_section += "**IMC 공식 분석**:\n"
new_section += "```\n"
new_section += "Kp = τ / (K * λ),  λ = λ_factor * τ\n"
new_section += "따라서: Kp = 1 / (K * λ_factor)  → τ에 무관!\n\n"
new_section += "Ki = 1 / (K * λ) = 1 / (K * λ_factor * τ)  → τ에 반비례!\n"
new_section += "```\n\n"
new_section += "**결과**:\n"
new_section += "1. ✅ **Kp는 τ와 무관** (변화 없음)\n"
new_section += "2. ✅ **Ki는 τ에 반비례** (τ가 1/50 → Ki는 50배)\n"
new_section += "3. ✅ **Ki/Kp = 1/τ = 0.5** (이론값과 일치!)\n\n"

new_section += "### 13.5 제어 성능 검증\n\n"
new_section += "**Arm_In 예시** (Kp=3.87, Ki=1.94):\n\n"
new_section += "| 각도 오차 | P 출력 | I 출력 (1초 후) | 총 출력 | 평가 |\n"
new_section += "|---|---|---|---|---|\n"
new_section += "| 10도 | 38.7% | 19.4% | 58.1% | ✅ 충분한 제어력 |\n"
new_section += "| 5도 | 19.4% | 9.7% | 29.1% | ✅ 적절한 제어 |\n"
new_section += "| 1도 | 3.9% | 1.9% | 5.8% | ✅ 미세 제어 가능 |\n"
new_section += "| 0.5도 | 1.9% | 1.0% | 2.9% | ✅ 정상 상태 오차 제거 |\n\n"

new_section += "### 13.6 적용 절차\n\n"
new_section += "**Step 1**: 안전 게인 (50% Ki)으로 시작\n"
new_section += "```\n"
new_section += "예: Arm_In → Kp=3.87, Ki=0.97\n"
new_section += "```\n\n"
new_section += "**Step 2**: 저속 테스트 (duty 40-50%)\n"
new_section += "- Step response 측정\n"
new_section += "- 오버슈트 < 10% 확인\n"
new_section += "- 정착시간 < 10초 확인\n\n"
new_section += "**Step 3**: Ki 점진 증가\n"
new_section += "- 60% → 80% → 100%\n"
new_section += "- 각 단계마다 안정성 확인\n\n"
new_section += "**Step 4**: 미세 조정\n"
new_section += "- 응답 느림 → Ki +20%\n"
new_section += "- 오버슈트 발생 → Ki -20%\n"
new_section += "- 정상상태 오차 → Ki +30%\n\n"

# Add FF lookup table if available
if ff_lookup_df is not None:
    new_section += "---\n\n"
    new_section += "## 📋 14. 속도 기반 FF 룩업 테이블 (상세)\n\n"
    new_section += "**각 축별로 속도 범위에 따라 FF 게인(Kv, K_offset)이 크게 변합니다.**\n\n"
    
    for axis_name, direction in axis_order:
        axis_key = f"{axis_name}_{direction}"
        df_axis = ff_lookup_df[ff_lookup_df['axis'] == axis_key]
        
        if len(df_axis) == 0:
            continue
        
        new_section += f"### 14.{axis_order.index((axis_name, direction)) + 1} {axis_name}_{direction}\n\n"
        new_section += "| 속도 범위 | Kv | K_offset | R² | 신뢰도 | 샘플 수 |\n"
        new_section += "|---|---|---|---|---|---|\n"
        
        for _, row in df_axis.iterrows():
            velocity_range = row['velocity_range']
            Kv = row['Kv']
            K_offset = row['K_offset']
            r2 = row['R2_ff']
            n_samples = int(row['n_samples'])
            
            # 신뢰도 판정
            if r2 >= 0.8 and n_samples >= 5:
                confidence = "✅ 높음"
            elif r2 >= 0.5 and n_samples >= 3:
                confidence = "⚠️ 중간"
            else:
                confidence = "🔴 낮음"
            
            new_section += f"| {velocity_range} | **{Kv:.3f}** | **{K_offset:.1f}** | {r2:.3f} | {confidence} | {n_samples} |\n"
        
        new_section += "\n"
    
    new_section += "### 14.7 구현 권장사항\n\n"
    new_section += "**신뢰도 높은 구간 우선 적용**:\n\n"
    new_section += "1. ✅ **Boom_Down**: 초저속, 저속 구간 (R²=0.88~0.99)\n"
    new_section += "2. ✅ **Boom_Up**: 초저속 구간 (R²=0.90)\n"
    new_section += "3. ✅ **Arm_In**: 초저속 구간 (R²=0.92)\n"
    new_section += "4. ⚠️ **나머지 축**: 실측 검증 필요\n\n"
    new_section += "**구현 방법**:\n"
    new_section += "```cpp\n"
    new_section += "// 속도 기반 FF 룩업 예시 (Boom_Down)\n"
    new_section += "float get_ff_kv(float velocity) {\n"
    new_section += "    if (abs(velocity) < 5.0) {\n"
    new_section += "        return 3.134;  // 초저속\n"
    new_section += "    } else if (abs(velocity) < 15.0) {\n"
    new_section += "        return 2.135;  // 저속\n"
    new_section += "    } else {\n"
    new_section += "        return 1.325;  // 중속 이상\n"
    new_section += "    }\n"
    new_section += "}\n"
    new_section += "```\n\n"

# Insert new section
parts = content.split(marker)
if len(parts) == 2:
    new_content = parts[0] + marker + new_section + "\n\n---\n\n" + parts[1]
else:
    print("오류: 마커를 기준으로 분할할 수 없습니다!")
    exit(1)

# Save updated file
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\n[OK] {summary_file} 업데이트 완료!")
print("\n추가된 섹션:")
print("  - 13. tau=2s 기준 최종 PID 게인")
if ff_lookup_df is not None:
    print("  - 14. 속도 기반 FF 룩업 테이블 (상세)")
print("\n" + "="*80)

