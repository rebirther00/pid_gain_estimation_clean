"""
Visualize Control Performance Analysis Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

# Setup Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
base = Path(__file__).parent.parent
error_dir = base / "output" / "error_analysis"
output_dir = error_dir

print("="*70)
print("Visualizing Performance Analysis")
print("="*70)

# Load summary
summary_file = error_dir / "performance_summary.csv"
df_summary = pd.read_csv(summary_file)

axes = df_summary['axis'].values

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color map for grades
grade_colors = {
    'Excellent': '#2ecc71',
    'Good': '#3498db',
    'Fair': '#f39c12'
}

# 1. Model Quality (R²) by Axis
ax1 = fig.add_subplot(gs[0, 0])
colors = [grade_colors[g] for g in df_summary['grade']]
bars = ax1.bar(range(len(axes)), df_summary['r_squared_mean'], color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (>0.9)')
ax1.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (>0.7)')
ax1.set_xticks(range(len(axes)))
ax1.set_xticklabels(axes, rotation=45, ha='right')
ax1.set_ylabel('R² (Model Fit)')
ax1.set_title('Model Quality (PID Tuning Basis)', fontweight='bold')
ax1.set_ylim([0.5, 1.0])
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=8)

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, df_summary['r_squared_mean'])):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
             ha='center', va='bottom', fontsize=8)

# 2. FF Quality (R²) by Axis
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(range(len(axes)), df_summary['ff_r_squared'], color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (>0.7)')
ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Fair (>0.5)')
ax2.set_xticks(range(len(axes)))
ax2.set_xticklabels(axes, rotation=45, ha='right')
ax2.set_ylabel('R² (FF Fit)')
ax2.set_title('FF Gain Quality (Velocity Control)', fontweight='bold')
ax2.set_ylim([0.0, 1.0])
ax2.grid(axis='y', alpha=0.3)
ax2.legend(fontsize=8)

for i, (bar, val) in enumerate(zip(bars, df_summary['ff_r_squared'])):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
             ha='center', va='bottom', fontsize=8)

# 3. Duty Prediction Error by Axis
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(range(len(axes)), df_summary['duty_error_%_mean'], color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (<10%)')
ax3.axhline(y=20, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (<20%)')
ax3.set_xticks(range(len(axes)))
ax3.set_xticklabels(axes, rotation=45, ha='right')
ax3.set_ylabel('Duty Error (%)')
ax3.set_title('FF Prediction Error', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.legend(fontsize=8)

for i, (bar, val) in enumerate(zip(bars, df_summary['duty_error_%_mean'])):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', 
             ha='center', va='bottom', fontsize=8)

# 4. Expected Settling Time
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar(range(len(axes)), df_summary['expected_settling'], color=colors, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(axes)))
ax4.set_xticklabels(axes, rotation=45, ha='right')
ax4.set_ylabel('Time (s)')
ax4.set_title('Expected Settling Time (4τ)', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, df_summary['expected_settling'])):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val:.0f}s', 
             ha='center', va='bottom', fontsize=8)

# 5. Performance Grade Distribution
ax5 = fig.add_subplot(gs[1, 1])
grade_counts = df_summary['grade'].value_counts()
grade_order = ['Excellent', 'Good', 'Fair']
grade_values = [grade_counts.get(g, 0) for g in grade_order]
grade_colors_list = [grade_colors[g] for g in grade_order]

bars = ax5.bar(grade_order, grade_values, color=grade_colors_list, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Number of Axes')
ax5.set_title('Performance Grade Distribution', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, grade_values):
    if val > 0:
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{int(val)}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Overall Metrics Radar Chart
ax6 = fig.add_subplot(gs[1, 2], projection='polar')

categories = ['Model\nQuality', 'FF\nQuality', 'Duty\nAccuracy', 'Response\nSpeed']
values = [
    df_summary['r_squared_mean'].mean(),
    df_summary['ff_r_squared'].mean(),
    1 - df_summary['duty_error_%_mean'].mean() / 100,  # Invert so higher is better
    1 - df_summary['expected_settling'].mean() / 400  # Normalize and invert
]

# Normalize to 0-1 range for radar
values = [max(0, min(1, v)) for v in values]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values += values[:1]  # Complete the circle
angles += angles[:1]

ax6.plot(angles, values, 'o-', linewidth=2, color='#3498db')
ax6.fill(angles, values, alpha=0.25, color='#3498db')
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontsize=9)
ax6.set_ylim(0, 1)
ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
ax6.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
ax6.set_title('Overall Performance Profile', fontweight='bold', pad=20)
ax6.grid(True)

# 7. Detailed comparison table
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('tight')
ax7.axis('off')

table_data = []
for _, row in df_summary.iterrows():
    table_data.append([
        row['axis'],
        row['grade'],
        f"{row['r_squared_mean']:.3f}",
        f"{row['ff_r_squared']:.3f}",
        f"{row['duty_error_%_mean']:.1f}%",
        f"{row['expected_settling']:.0f}s",
        f"{row['n_samples']}"
    ])

table = ax7.table(
    cellText=table_data,
    colLabels=['Axis', 'Grade', 'Model R²', 'FF R²', 'Duty Err', 'Settling', 'Samples'],
    cellLoc='center',
    loc='center',
    colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color rows by grade
for i, row in enumerate(df_summary.itertuples(), start=1):
    color = grade_colors[row.grade]
    for j in range(7):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.3)

# Header styling
for j in range(7):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax7.set_title('Detailed Performance Comparison', fontweight='bold', fontsize=12, pad=20)

plt.suptitle('V3 Gains: Expected Control Performance Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_dir / 'performance_analysis_overview.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'performance_analysis_overview.png'}")

# Individual axis plots
for axis in axes:
    details_file = error_dir / f"{axis}_performance_details.csv"
    if not details_file.exists():
        continue
    
    df_details = pd.read_csv(details_file)
    
    fig, axes_sub = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{axis} - Detailed Performance Analysis', fontsize=14, fontweight='bold')
    
    # 1. R² by Duty
    ax = axes_sub[0, 0]
    duty_r2 = df_details.groupby('duty_actual')['r_squared'].agg(['mean', 'std', 'count'])
    ax.errorbar(duty_r2.index, duty_r2['mean'], yerr=duty_r2['std'], 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good')
    ax.set_xlabel('Duty (%)')
    ax.set_ylabel('R² (Model Fit)')
    ax.set_title('Model Quality vs Duty', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Duty Error by Duty
    ax = axes_sub[0, 1]
    duty_err = df_details.groupby('duty_actual')['duty_error_%'].agg(['mean', 'std'])
    ax.errorbar(duty_err.index, duty_err['mean'], yerr=duty_err['std'], 
                marker='s', capsize=5, linewidth=2, markersize=8, color='coral')
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Excellent (<10%)')
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Good (<20%)')
    ax.set_xlabel('Duty (%)')
    ax.set_ylabel('Duty Prediction Error (%)')
    ax.set_title('FF Accuracy vs Duty', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Duty Predicted vs Actual
    ax = axes_sub[1, 0]
    ax.scatter(df_details['duty_actual'], df_details['duty_predicted'], 
               c=df_details['r_squared'], cmap='viridis', s=50, alpha=0.6)
    lim_min = min(df_details['duty_actual'].min(), df_details['duty_predicted'].min())
    lim_max = max(df_details['duty_actual'].max(), df_details['duty_predicted'].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual Duty (%)')
    ax.set_ylabel('Predicted Duty (FF) (%)')
    ax.set_title('FF Prediction Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('R² (Model Quality)')
    
    # 4. Distribution of Errors
    ax = axes_sub[1, 1]
    ax.hist(df_details['duty_error_%'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=df_details['duty_error_%'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f"Mean: {df_details['duty_error_%'].mean():.1f}%")
    ax.set_xlabel('Duty Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Duty Error Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{axis}_detailed_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{axis}_detailed_analysis.png'}")

plt.close('all')

print("\n" + "="*70)
print("Visualization Complete")
print("="*70)
print(f"\nAll plots saved in: {output_dir}/")

