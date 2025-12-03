"""
Exploratory Data Analysis (EDA) for Student Performance Dataset.

This script focuses on analyzing the Pass/Fail labeling strategy and threshold visualizations:
1. GPA distribution by PassFail
2. AvgTestScore distribution by PassFail
3. AttendanceRate vs AvgTestScore scatter plot colored by PassFail
4. Threshold visualizations for GPA, AvgTestScore, and AttendanceRate

Note: This EDA should be run AFTER data preprocessing (prepare_data.py).
Works in both local and SageMaker Processing environments.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot (required for SageMaker headless mode)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from argparse import ArgumentParser
import warnings
import os
warnings.filterwarnings('ignore')

# Threshold values used in labeling
GPA_THRESHOLD = 2.0
AVG_TEST_SCORE_THRESHOLD = 73.0
ATTENDANCE_RATE_THRESHOLD = 0.85

def load_processed_data(split: str = "train", input_data_dir: str = None):
    """Load processed dataset (must run prepare_data.py first).
    
    Args:
        split: Dataset split to load (train, validation, test)
        input_data_dir: Directory containing processed data (default: auto-detect local or SageMaker)
    
    Returns:
        DataFrame with processed data
    """
    # Auto-detect SageMaker environment
    if input_data_dir is None:
        if os.path.exists("/opt/ml/processing/input"):
            input_data_dir = "/opt/ml/processing/input"
            print("[INFO] Detected SageMaker Processing environment")
        else:
            input_data_dir = "data/processed"
            print("[INFO] Using local data directory")
    
    input_path = Path(input_data_dir)
    file_path = input_path / f"{split}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}\n"
            "Please run 'python scripts/prepare_data.py' first to generate processed data."
        )
    
    print(f"Loading processed {split} dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Verify pass_fail column exists
    if "pass_fail" not in df.columns:
        raise ValueError(
            "Column 'pass_fail' not found in processed data.\n"
            "Please run 'python scripts/prepare_data.py' first to create labels."
        )
    
    return df

def compute_avg_test_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average test score from individual test scores."""
    df = df.copy()
    
    # Check if AvgTestScore already exists
    if "AvgTestScore" in df.columns:
        return df
    
    # Check if individual test scores exist
    test_cols = ["TestScore_Math", "TestScore_Reading", "TestScore_Science"]
    if all(col in df.columns for col in test_cols):
        df["AvgTestScore"] = (
            df["TestScore_Math"] + 
            df["TestScore_Reading"] + 
            df["TestScore_Science"]
        ) / 3.0
        print("[OK] Computed AvgTestScore from individual test scores")
    else:
        raise ValueError(
            "Cannot compute AvgTestScore: Missing test score columns.\n"
            "Required columns: TestScore_Math, TestScore_Reading, TestScore_Science"
        )
    
    return df

def plot_gpa_distribution_by_passfail(df: pd.DataFrame, output_path: Path):
    """Plot 1: GPA distribution by PassFail (KDE and histogram)."""
    print("\n" + "=" * 60)
    print("Plot 1: GPA Distribution by PassFail")
    print("=" * 60)
    
    if "GPA" not in df.columns:
        print("[WARNING] GPA column not found - skipping this plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate Pass and Fail
    pass_data = df[df["pass_fail"] == 1]["GPA"]
    fail_data = df[df["pass_fail"] == 0]["GPA"]
    
    # Left plot: Histogram
    axes[0].hist(fail_data, bins=50, alpha=0.6, label=f'Fail (n={len(fail_data):,})', 
                 color='red', edgecolor='black', density=True)
    axes[0].hist(pass_data, bins=50, alpha=0.6, label=f'Pass (n={len(pass_data):,})', 
                 color='green', edgecolor='black', density=True)
    axes[0].axvline(GPA_THRESHOLD, color='blue', linestyle='--', linewidth=2, 
                    label=f'Threshold: {GPA_THRESHOLD}')
    axes[0].set_xlabel('GPA', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('GPA Distribution by PassFail (Histogram)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: KDE
    sns.kdeplot(data=df, x="GPA", hue="pass_fail", ax=axes[1], 
                palette={0: 'red', 1: 'green'}, fill=True, alpha=0.6)
    axes[1].axvline(GPA_THRESHOLD, color='blue', linestyle='--', linewidth=2, 
                    label=f'Threshold: {GPA_THRESHOLD}')
    axes[1].set_xlabel('GPA', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('GPA Distribution by PassFail (KDE)', fontsize=13, fontweight='bold')
    axes[1].legend(['Fail', 'Pass', f'Threshold: {GPA_THRESHOLD}'], fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "gpa_distribution_by_passfail.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: gpa_distribution_by_passfail.png")
    
    # Print statistics
    print(f"\nGPA Statistics by PassFail:")
    print(f"  Pass (GPA >= {GPA_THRESHOLD}): Mean={pass_data.mean():.2f}, Median={pass_data.median():.2f}")
    print(f"  Fail (GPA < {GPA_THRESHOLD} or other conditions): Mean={fail_data.mean():.2f}, Median={fail_data.median():.2f}")

def plot_avgtestscore_distribution_by_passfail(df: pd.DataFrame, output_path: Path):
    """Plot 2: AvgTestScore distribution by PassFail (KDE and histogram)."""
    print("\n" + "=" * 60)
    print("Plot 2: AvgTestScore Distribution by PassFail")
    print("=" * 60)
    
    if "AvgTestScore" not in df.columns:
        print("[WARNING] AvgTestScore column not found - skipping this plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate Pass and Fail
    pass_data = df[df["pass_fail"] == 1]["AvgTestScore"]
    fail_data = df[df["pass_fail"] == 0]["AvgTestScore"]
    
    # Left plot: Histogram
    axes[0].hist(fail_data, bins=50, alpha=0.6, label=f'Fail (n={len(fail_data):,})', 
                 color='red', edgecolor='black', density=True)
    axes[0].hist(pass_data, bins=50, alpha=0.6, label=f'Pass (n={len(pass_data):,})', 
                 color='green', edgecolor='black', density=True)
    axes[0].axvline(AVG_TEST_SCORE_THRESHOLD, color='blue', linestyle='--', linewidth=2, 
                    label=f'Threshold: {AVG_TEST_SCORE_THRESHOLD}')
    axes[0].set_xlabel('Average Test Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('AvgTestScore Distribution by PassFail (Histogram)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: KDE
    sns.kdeplot(data=df, x="AvgTestScore", hue="pass_fail", ax=axes[1], 
                palette={0: 'red', 1: 'green'}, fill=True, alpha=0.6)
    axes[1].axvline(AVG_TEST_SCORE_THRESHOLD, color='blue', linestyle='--', linewidth=2, 
                    label=f'Threshold: {AVG_TEST_SCORE_THRESHOLD}')
    axes[1].set_xlabel('Average Test Score', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('AvgTestScore Distribution by PassFail (KDE)', fontsize=13, fontweight='bold')
    axes[1].legend(['Fail', 'Pass', f'Threshold: {AVG_TEST_SCORE_THRESHOLD}'], fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "avgtestscore_distribution_by_passfail.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: avgtestscore_distribution_by_passfail.png")
    
    # Print statistics
    print(f"\nAvgTestScore Statistics by PassFail:")
    print(f"  Pass (AvgTestScore >= {AVG_TEST_SCORE_THRESHOLD}): Mean={pass_data.mean():.2f}, Median={pass_data.median():.2f}")
    print(f"  Fail (AvgTestScore < {AVG_TEST_SCORE_THRESHOLD} or other conditions): Mean={fail_data.mean():.2f}, Median={fail_data.median():.2f}")

def plot_attendance_vs_testscore_scatter(df: pd.DataFrame, output_path: Path):
    """Plot 3: AttendanceRate vs AvgTestScore scatter plot colored by PassFail."""
    print("\n" + "=" * 60)
    print("Plot 3: AttendanceRate vs AvgTestScore Scatter Plot")
    print("=" * 60)
    
    required_cols = ["AttendanceRate", "AvgTestScore", "pass_fail"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARNING] Missing columns: {missing_cols} - skipping this plot")
        return
    
    # Sample for faster plotting if dataset is large
    if len(df) > 50000:
        plot_df = df.sample(n=50000, random_state=42)
        print(f"  Using subsample of {len(plot_df):,} points for visualization")
    else:
        plot_df = df
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separate Pass and Fail for better visualization
    pass_df = plot_df[plot_df["pass_fail"] == 1]
    fail_df = plot_df[plot_df["pass_fail"] == 0]
    
    # Scatter plot
    ax.scatter(fail_df["AttendanceRate"], fail_df["AvgTestScore"], 
               alpha=0.5, s=10, c='red', label=f'Fail (n={len(fail_df):,})', edgecolors='none')
    ax.scatter(pass_df["AttendanceRate"], pass_df["AvgTestScore"], 
               alpha=0.5, s=10, c='green', label=f'Pass (n={len(pass_df):,})', edgecolors='none')
    
    # Draw threshold lines
    ax.axhline(AVG_TEST_SCORE_THRESHOLD, color='blue', linestyle='--', linewidth=2, 
               label=f'AvgTestScore Threshold: {AVG_TEST_SCORE_THRESHOLD}')
    ax.axvline(ATTENDANCE_RATE_THRESHOLD, color='blue', linestyle='--', linewidth=2, 
               label=f'AttendanceRate Threshold: {ATTENDANCE_RATE_THRESHOLD}')
    
    # Shade the "Pass region" (top right quadrant above both thresholds)
    ax.axhspan(AVG_TEST_SCORE_THRESHOLD, plot_df["AvgTestScore"].max() + 5, 
               xmin=(ATTENDANCE_RATE_THRESHOLD - plot_df["AttendanceRate"].min()) / 
                    (plot_df["AttendanceRate"].max() - plot_df["AttendanceRate"].min()),
               xmax=1, alpha=0.1, color='green', label='Pass Region')
    
    ax.set_xlabel('Attendance Rate', fontsize=12)
    ax.set_ylabel('Average Test Score', fontsize=12)
    ax.set_title('AttendanceRate vs AvgTestScore (Colored by PassFail)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "attendance_vs_testscore_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: attendance_vs_testscore_scatter.png")
    
    # Print statistics
    print(f"\nPass Region Analysis:")
    pass_region = df[(df["AvgTestScore"] >= AVG_TEST_SCORE_THRESHOLD) & 
                      (df["AttendanceRate"] >= ATTENDANCE_RATE_THRESHOLD)]
    print(f"  Students in Pass region (both thresholds met): {len(pass_region):,} ({len(pass_region)/len(df)*100:.2f}%)")
    print(f"  Of these, actually labeled Pass: {(pass_region['pass_fail']==1).sum():,} ({(pass_region['pass_fail']==1).sum()/len(pass_region)*100:.2f}%)")

def plot_threshold_visualizations(df: pd.DataFrame, output_path: Path):
    """Plot 4: Three threshold visualizations (GPA, AvgTestScore, AttendanceRate)."""
    print("\n" + "=" * 60)
    print("Plot 4: Threshold Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: GPA distribution with threshold
    if "GPA" in df.columns:
        axes[0].hist(df["GPA"], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(GPA_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                        label=f'Threshold: {GPA_THRESHOLD}')
        axes[0].set_xlabel('GPA', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'GPA Distribution\n(Threshold: {GPA_THRESHOLD})', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics text
        above_threshold = (df["GPA"] >= GPA_THRESHOLD).sum()
        below_threshold = (df["GPA"] < GPA_THRESHOLD).sum()
        axes[0].text(0.05, 0.95, f'Above: {above_threshold:,} ({above_threshold/len(df)*100:.1f}%)\n'
                                  f'Below: {below_threshold:,} ({below_threshold/len(df)*100:.1f}%)',
                     transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[0].text(0.5, 0.5, 'GPA column not found', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('GPA Distribution', fontsize=13, fontweight='bold')
    
    # Plot 2: AvgTestScore distribution with threshold
    if "AvgTestScore" in df.columns:
        axes[1].hist(df["AvgTestScore"], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].axvline(AVG_TEST_SCORE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                        label=f'Threshold: {AVG_TEST_SCORE_THRESHOLD}')
        axes[1].set_xlabel('Average Test Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'AvgTestScore Distribution\n(Threshold: {AVG_TEST_SCORE_THRESHOLD})', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        above_threshold = (df["AvgTestScore"] >= AVG_TEST_SCORE_THRESHOLD).sum()
        below_threshold = (df["AvgTestScore"] < AVG_TEST_SCORE_THRESHOLD).sum()
        axes[1].text(0.05, 0.95, f'Above: {above_threshold:,} ({above_threshold/len(df)*100:.1f}%)\n'
                                  f'Below: {below_threshold:,} ({below_threshold/len(df)*100:.1f}%)',
                     transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[1].text(0.5, 0.5, 'AvgTestScore column not found', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('AvgTestScore Distribution', fontsize=13, fontweight='bold')
    
    # Plot 3: AttendanceRate distribution with threshold
    if "AttendanceRate" in df.columns:
        axes[2].hist(df["AttendanceRate"], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[2].axvline(ATTENDANCE_RATE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                        label=f'Threshold: {ATTENDANCE_RATE_THRESHOLD}')
        axes[2].set_xlabel('Attendance Rate', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title(f'AttendanceRate Distribution\n(Threshold: {ATTENDANCE_RATE_THRESHOLD})', 
                         fontsize=13, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Add statistics text
        above_threshold = (df["AttendanceRate"] >= ATTENDANCE_RATE_THRESHOLD).sum()
        below_threshold = (df["AttendanceRate"] < ATTENDANCE_RATE_THRESHOLD).sum()
        axes[2].text(0.05, 0.95, f'Above: {above_threshold:,} ({above_threshold/len(df)*100:.1f}%)\n'
                                  f'Below: {below_threshold:,} ({below_threshold/len(df)*100:.1f}%)',
                     transform=axes[2].transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[2].text(0.5, 0.5, 'AttendanceRate column not found', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('AttendanceRate Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "threshold_visualizations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: threshold_visualizations.png")
    
    # Print summary statistics
    print(f"\nThreshold Summary Statistics:")
    if "GPA" in df.columns:
        gpa_above = (df["GPA"] >= GPA_THRESHOLD).sum()
        print(f"  GPA >= {GPA_THRESHOLD}: {gpa_above:,} ({gpa_above/len(df)*100:.2f}%)")
    if "AvgTestScore" in df.columns:
        test_above = (df["AvgTestScore"] >= AVG_TEST_SCORE_THRESHOLD).sum()
        print(f"  AvgTestScore >= {AVG_TEST_SCORE_THRESHOLD}: {test_above:,} ({test_above/len(df)*100:.2f}%)")
    if "AttendanceRate" in df.columns:
        attend_above = (df["AttendanceRate"] >= ATTENDANCE_RATE_THRESHOLD).sum()
        print(f"  AttendanceRate >= {ATTENDANCE_RATE_THRESHOLD}: {attend_above:,} ({attend_above/len(df)*100:.2f}%)")

def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate summary report with key statistics."""
    print("\n" + "=" * 60)
    print("Generating Summary Report")
    print("=" * 60)
    
    report = {
        "dataset_info": {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns))
        },
        "pass_fail_distribution": {
            "pass_count": int((df["pass_fail"] == 1).sum()),
            "fail_count": int((df["pass_fail"] == 0).sum()),
            "pass_percentage": float((df["pass_fail"] == 1).sum() / len(df) * 100),
            "fail_percentage": float((df["pass_fail"] == 0).sum() / len(df) * 100)
        },
        "thresholds": {
            "gpa_threshold": GPA_THRESHOLD,
            "avg_test_score_threshold": AVG_TEST_SCORE_THRESHOLD,
            "attendance_rate_threshold": ATTENDANCE_RATE_THRESHOLD
        }
    }
    
    # Add statistics for each threshold variable
    if "GPA" in df.columns:
        report["gpa_statistics"] = {
            "mean": float(df["GPA"].mean()),
            "median": float(df["GPA"].median()),
            "std": float(df["GPA"].std()),
            "min": float(df["GPA"].min()),
            "max": float(df["GPA"].max()),
            "above_threshold": int((df["GPA"] >= GPA_THRESHOLD).sum()),
            "below_threshold": int((df["GPA"] < GPA_THRESHOLD).sum())
        }
    
    if "AvgTestScore" in df.columns:
        report["avgtestscore_statistics"] = {
            "mean": float(df["AvgTestScore"].mean()),
            "median": float(df["AvgTestScore"].median()),
            "std": float(df["AvgTestScore"].std()),
            "min": float(df["AvgTestScore"].min()),
            "max": float(df["AvgTestScore"].max()),
            "above_threshold": int((df["AvgTestScore"] >= AVG_TEST_SCORE_THRESHOLD).sum()),
            "below_threshold": int((df["AvgTestScore"] < AVG_TEST_SCORE_THRESHOLD).sum())
        }
    
    if "AttendanceRate" in df.columns:
        report["attendancerate_statistics"] = {
            "mean": float(df["AttendanceRate"].mean()),
            "median": float(df["AttendanceRate"].median()),
            "std": float(df["AttendanceRate"].std()),
            "min": float(df["AttendanceRate"].min()),
            "max": float(df["AttendanceRate"].max()),
            "above_threshold": int((df["AttendanceRate"] >= ATTENDANCE_RATE_THRESHOLD).sum()),
            "below_threshold": int((df["AttendanceRate"] < ATTENDANCE_RATE_THRESHOLD).sum())
        }
    
    # Save report
    report_path = output_path / "eda_summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"[OK] Saved summary report to: {report_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EDA Summary")
    print(f"{'='*60}")
    print(f"Total rows: {report['dataset_info']['total_rows']:,}")
    print(f"\nPass/Fail Distribution:")
    print(f"  Pass: {report['pass_fail_distribution']['pass_count']:,} ({report['pass_fail_distribution']['pass_percentage']:.2f}%)")
    print(f"  Fail: {report['pass_fail_distribution']['fail_count']:,} ({report['pass_fail_distribution']['fail_percentage']:.2f}%)")
    print(f"\nLabeling Thresholds:")
    print(f"  GPA >= {GPA_THRESHOLD}")
    print(f"  AvgTestScore >= {AVG_TEST_SCORE_THRESHOLD}")
    print(f"  AttendanceRate >= {ATTENDANCE_RATE_THRESHOLD}")
    print(f"  (All three conditions must be true for Pass)")

def main():
    parser = ArgumentParser(
        description="EDA for Student Performance Dataset - Focus on Pass/Fail Labeling Analysis"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to analyze (default: train). Note: Must run prepare_data.py first!"
    )
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        help="Directory containing processed data files (default: auto-detect local or SageMaker)"
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=None,
        help="Directory to save EDA results (default: results/eda for local, /opt/ml/processing/output for SageMaker)"
    )
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_data_dir is None:
        if os.path.exists("/opt/ml/processing/output"):
            output_dir = Path("/opt/ml/processing/output")
            print("[INFO] Detected SageMaker Processing environment - using /opt/ml/processing/output")
        else:
            output_dir = Path("results/eda")
            print("[INFO] Using local output directory: results/eda")
    else:
        output_dir = Path(args.output_data_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Student Performance Dataset - Exploratory Data Analysis")
    print("=" * 60)
    print("\nNote: This EDA analyzes processed data with Pass/Fail labels.")
    print("      Make sure you have run 'python scripts/prepare_data.py' first!")
    print()
    
    # Load processed data
    df = load_processed_data(args.split, args.input_data_dir)
    
    # Compute AvgTestScore if needed
    df = compute_avg_test_score(df)
    
    # Generate all visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Plot 1: GPA distribution by PassFail
    plot_gpa_distribution_by_passfail(df, output_dir)
    
    # Plot 2: AvgTestScore distribution by PassFail
    plot_avgtestscore_distribution_by_passfail(df, output_dir)
    
    # Plot 3: AttendanceRate vs AvgTestScore scatter
    plot_attendance_vs_testscore_scatter(df, output_dir)
    
    # Plot 4: Threshold visualizations
    plot_threshold_visualizations(df, output_dir)
    
    # Generate summary report
    generate_summary_report(df, output_dir)
    
    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. gpa_distribution_by_passfail.png")
    print("  2. avgtestscore_distribution_by_passfail.png")
    print("  3. attendance_vs_testscore_scatter.png")
    print("  4. threshold_visualizations.png")
    print("  5. eda_summary_report.json")
    print("\nNext steps:")
    print("  1. Review EDA visualizations and summary report")
    print("  2. Train models: python scripts/train.py")
    print("  3. Start API: python serve.py")

if __name__ == "__main__":
    main()
