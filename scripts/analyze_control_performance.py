"""
Control Performance Analysis
Evaluates expected performance of V3 gains through:
1. Model fit quality (R²)
2. FF gain accuracy (duty vs velocity correlation)
3. Duty-based performance breakdown
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Paths
base = Path(__file__).parent.parent
post_dir = base / "output" / "post_process_v3"
integrated_dir = base / "output" / "integrated_v3"
output_dir = base / "output" / "error_analysis"
output_dir.mkdir(exist_ok=True)

print("="*70)
print("Control Performance Analysis")
print("="*70)
print("\nEvaluating V3 gains through model quality and FF accuracy\n")

# Load final gains
final_gains_file = post_dir / "final_gains.json"
with open(final_gains_file, 'r') as f:
    final_gains = json.load(f)

# Load file statistics
stats_file = integrated_dir / "debug" / "file_statistics.csv"
stats_df = pd.read_csv(stats_file)

axes = ['Arm_In', 'Arm_Out', 'Boom_Up', 'Boom_Down', 'Bucket_In', 'Bucket_Out']
all_results = []

for axis in axes:
    print(f"\n{axis}:")
    print("-"*70)
    
    # Load individual gains
    gains_file = post_dir / f"{axis}_individual_gains.csv"
    if not gains_file.exists():
        continue
    
    df_gains = pd.read_csv(gains_file)
    
    # Merge with velocities
    df_merged = df_gains.merge(
        stats_df[stats_df['group'] == axis][['file', 'duty', 'velocity']], 
        on='file',
        how='inner',
        suffixes=('', '_actual')
    )
    
    print(f"  Total samples: {len(df_merged)}")
    
    # Filter valid samples
    df_valid = df_merged[
        (~df_merged.get('low_quality', False)) & 
        (df_merged['r_squared'] >= 0.5)
    ].copy()
    
    print(f"  Valid samples: {len(df_valid)} (R² >= 0.5)")
    
    if len(df_valid) == 0:
        continue
    
    # Get final gains
    fg = final_gains[axis]['final_gains']
    ff = final_gains[axis]['ff_gain']
    
    Kp_final = fg['Kp']
    Ki_final = fg['Ki']
    Kv_final = ff['Kv']
    K_offset = ff.get('K_offset', 0)
    r_squared_ff = ff.get('r_squared', 0)
    
    # 1. Model Quality Analysis
    print(f"\n  1. Model Quality (PID tuning basis):")
    print(f"     R² mean:  {df_valid['r_squared'].mean():.3f}")
    print(f"     R² min:   {df_valid['r_squared'].min():.3f}")
    print(f"     R² std:   {df_valid['r_squared'].std():.3f}")
    
    excellent = len(df_valid[df_valid['r_squared'] > 0.9])
    good = len(df_valid[(df_valid['r_squared'] > 0.7) & (df_valid['r_squared'] <= 0.9)])
    fair = len(df_valid[df_valid['r_squared'] <= 0.7])
    
    print(f"     Excellent (R²>0.9): {excellent}/{len(df_valid)} ({excellent/len(df_valid)*100:.0f}%)")
    print(f"     Good (0.7<R²≤0.9):  {good}/{len(df_valid)} ({good/len(df_valid)*100:.0f}%)")
    print(f"     Fair (R²≤0.7):      {fair}/{len(df_valid)} ({fair/len(df_valid)*100:.0f}%)")
    
    # 2. FF Gain Accuracy
    print(f"\n  2. FF Gain Accuracy (velocity control):")
    print(f"     Kv:        {Kv_final:.3f}")
    print(f"     K_offset:  {K_offset:.1f}")
    print(f"     FF R²:     {r_squared_ff:.3f}")
    
    # Calculate FF prediction error
    df_valid['duty_predicted'] = Kv_final * df_valid['velocity'] + K_offset
    df_valid['duty_error'] = abs(df_valid['duty_predicted'] - df_valid['duty_actual'])
    df_valid['duty_error_%'] = df_valid['duty_error'] / df_valid['duty_actual'] * 100
    
    print(f"     Duty error (mean):  {df_valid['duty_error'].mean():.1f}% ({df_valid['duty_error_%'].mean():.1f}%)")
    print(f"     Duty error (max):   {df_valid['duty_error'].max():.1f}% ({df_valid['duty_error_%'].max():.1f}%)")
    
    # 3. Performance by Duty Level
    print(f"\n  3. Performance by Duty Level:")
    
    duty_performance = []
    for duty in sorted(df_valid['duty_actual'].unique()):
        duty_data = df_valid[df_valid['duty_actual'] == duty]
        if len(duty_data) == 0:
            continue
        
        r2_mean = duty_data['r_squared'].mean()
        duty_err_mean = duty_data['duty_error_%'].mean()
        n = len(duty_data)
        
        duty_performance.append({
            'duty': duty,
            'n': n,
            'r_squared': r2_mean,
            'duty_error_%': duty_err_mean
        })
        
        print(f"     Duty {duty:3d}%: R²={r2_mean:.3f}, Duty err={duty_err_mean:.1f}% (n={n})")
    
    # Overall assessment
    print(f"\n  Overall Assessment:")
    
    if df_valid['r_squared'].mean() > 0.9 and r_squared_ff > 0.7:
        grade = "Excellent"
        emoji = "Excellent"
    elif df_valid['r_squared'].mean() > 0.7 and r_squared_ff > 0.5:
        grade = "Good"
        emoji = "Good"
    else:
        grade = "Fair"
        emoji = "Fair"
    
    print(f"     Grade: {grade}")
    print(f"     PID basis: {df_valid['r_squared'].mean():.3f} (model fit)")
    print(f"     FF basis:  {r_squared_ff:.3f} (velocity correlation)")
    
    # Expected control characteristics
    print(f"\n  Expected Control Characteristics:")
    
    # Response speed (based on tau)
    avg_tau = df_valid['tau'].mean()
    print(f"     Avg time constant: {avg_tau:.2f}s")
    print(f"     Expected settling: {4*avg_tau:.2f}s (4*tau)")
    
    # Stability (based on Kp range)
    kp_individual_std = df_valid['Kp'].std()
    kp_variation = kp_individual_std / Kp_final * 100
    print(f"     Kp consistency: {kp_variation:.1f}% variation")
    
    if kp_variation < 30:
        print(f"       -> Low variation, stable across conditions")
    elif kp_variation < 50:
        print(f"       -> Moderate variation, generally stable")
    else:
        print(f"       -> High variation, may need tuning")
    
    # Save details
    df_valid[['file', 'duty_actual', 'velocity', 'r_squared', 'Kp', 'Ki', 'tau', 
              'duty_predicted', 'duty_error', 'duty_error_%']].to_csv(
        output_dir / f"{axis}_performance_details.csv", index=False
    )
    
    # Summary
    all_results.append({
        'axis': axis,
        'n_samples': len(df_valid),
        'r_squared_mean': df_valid['r_squared'].mean(),
        'r_squared_min': df_valid['r_squared'].min(),
        'ff_r_squared': r_squared_ff,
        'duty_error_%_mean': df_valid['duty_error_%'].mean(),
        'duty_error_%_max': df_valid['duty_error_%'].max(),
        'avg_tau': avg_tau,
        'expected_settling': 4*avg_tau,
        'kp_variation_%': kp_variation,
        'grade': grade
    })

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)

if all_results:
    df_summary = pd.DataFrame(all_results)
    
    print("\n")
    print(df_summary[['axis', 'grade', 'r_squared_mean', 'ff_r_squared', 
                      'duty_error_%_mean', 'expected_settling']].to_string(index=False))
    
    summary_file = output_dir / "performance_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    
    # Overall
    print("\n" + "="*70)
    print("Overall Assessment")
    print("="*70)
    
    excellent_count = len(df_summary[df_summary['grade'] == 'Excellent'])
    good_count = len(df_summary[df_summary['grade'] == 'Good'])
    fair_count = len(df_summary[df_summary['grade'] == 'Fair'])
    
    print(f"\nPerformance Distribution:")
    print(f"  Excellent: {excellent_count}/{len(df_summary)} axes")
    print(f"  Good:      {good_count}/{len(df_summary)} axes")
    print(f"  Fair:      {fair_count}/{len(df_summary)} axes")
    
    print(f"\nAverage Metrics:")
    print(f"  Model quality (R²):     {df_summary['r_squared_mean'].mean():.3f}")
    print(f"  FF quality (R²):        {df_summary['ff_r_squared'].mean():.3f}")
    print(f"  Duty prediction error:  {df_summary['duty_error_%_mean'].mean():.1f}%")
    print(f"  Expected settling time: {df_summary['expected_settling'].mean():.2f}s")
    
    print(f"\nConclusion:")
    if excellent_count >= len(df_summary) * 0.7:
        print("  V3 gains are expected to provide EXCELLENT control performance")
    elif good_count + excellent_count >= len(df_summary) * 0.8:
        print("  V3 gains are expected to provide GOOD control performance")
    else:
        print("  V3 gains should provide ACCEPTABLE control performance")
        print("  Fine-tuning may be needed for optimal results")
    
    print(f"\nKey Strengths:")
    if df_summary['r_squared_mean'].mean() > 0.9:
        print(f"  - Excellent model fit (R² = {df_summary['r_squared_mean'].mean():.3f})")
    if df_summary['ff_r_squared'].mean() > 0.7:
        print(f"  - Strong FF correlation (R² = {df_summary['ff_r_squared'].mean():.3f})")
    if df_summary['duty_error_%_mean'].mean() < 15:
        print(f"  - Accurate velocity control ({df_summary['duty_error_%_mean'].mean():.1f}% error)")
    
    print(f"\nRecommendations:")
    print("  1. Use V3 gains as starting point")
    print("  2. Fine-tune in actual system if needed")
    print("  3. Monitor performance across full duty range")
    
    # Save overall summary
    with open(output_dir / "overall_assessment.txt", 'w') as f:
        f.write("V3 GAINS PERFORMANCE ASSESSMENT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model Quality (R²):      {df_summary['r_squared_mean'].mean():.3f}\n")
        f.write(f"FF Quality (R²):         {df_summary['ff_r_squared'].mean():.3f}\n")
        f.write(f"Duty Prediction Error:   {df_summary['duty_error_%_mean'].mean():.1f}%\n")
        f.write(f"Expected Settling Time:  {df_summary['expected_settling'].mean():.2f}s\n")
        f.write(f"\nPerformance Grade: ")
        if excellent_count >= len(df_summary) * 0.7:
            f.write("EXCELLENT\n")
        elif good_count + excellent_count >= len(df_summary) * 0.8:
            f.write("GOOD\n")
        else:
            f.write("ACCEPTABLE\n")

print("\n" + "="*70)
print("Analysis Complete")
print("="*70)
print(f"\nResults saved in: {output_dir}/")

