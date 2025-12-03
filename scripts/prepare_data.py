"""
Data preparation script for Student Performance Dataset.
Loads train/validation/test splits, creates at-risk vs good academic standing target 
using GPA + test-score labeling, and applies preprocessing.

Labeling strategy: Pass requires GPA >= 2.0 AND AvgTestScore >= 73.
All other students are labeled as Fail (at-risk). Attendance is kept as a predictive feature only.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Clean feature list - excludes GPA and test scores (to prevent data leakage)
# Test scores are directly tied to GPA, so they would make prediction trivial
CLEAN_FEATURE_LIST = [
    # Behavioral and lifestyle / study characteristics
    "AttendanceRate",
    "StudyHours",
    "InternetAccess",
    "Extracurricular",
    "PartTimeJob",
    "ParentSupport",
    "Romantic",
    "FreeTime",
    "GoOut",
    # Demographic and context (excluding protected attributes)
    "SES_Quartile",
    "ParentalEducation",
    "SchoolType",
    "Locale",
    # Optional demographic features (keeping them as they provide context)
    "Age",
    "Grade"
]

def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and convert data types to ensure consistency.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        DataFrame with validated data types
    """
    df = df.copy()
    
    # Expected data types
    type_mapping = {
        "Age": "int64",
        "Grade": "int64",
        "SES_Quartile": "int64",
        "InternetAccess": "int64",
        "Extracurricular": "int64",
        "PartTimeJob": "int64",
        "ParentSupport": "int64",
        "Romantic": "int64",
        "FreeTime": "int64",
        "GoOut": "int64"
    }
    
    for col, expected_type in type_mapping.items():
        if col in df.columns:
            try:
                if expected_type == "int64":
                    # Convert to numeric, handle NaN, then convert to int
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with mode or 0 before converting to int
                    if df[col].isnull().any():
                        fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                        df[col] = df[col].fillna(fill_value)
                    df[col] = df[col].astype('int64')
            except Exception as e:
                print(f"  [WARNING] Could not convert {col} to {expected_type}: {e}")
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: DataFrame with potential missing values
        strategy: Strategy for handling missing values ('drop', 'forward_fill', 'mean', 'median')
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        return df
    
    print(f"  Handling {missing_before:,} missing values using strategy: {strategy}")
    
    if strategy == "drop":
        # Drop rows with any missing values in critical columns
        critical_cols = ["GPA", "AttendanceRate", "StudyHours"] + CLEAN_FEATURE_LIST
        critical_cols = [col for col in critical_cols if col in df.columns]
        initial_rows = len(df)
        df = df.dropna(subset=critical_cols)
        dropped = initial_rows - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped:,} rows with missing critical values")
    
    elif strategy == "forward_fill":
        df = df.ffill()
    
    elif strategy == "mean":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    elif strategy == "median":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    missing_after = df.isnull().sum().sum()
    if missing_after > 0:
        print(f"  [WARNING] {missing_after:,} missing values remain")
    
    return df

def validate_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clip feature values to expected ranges.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        DataFrame with validated ranges
    """
    df = df.copy()
    
    # Expected ranges for numerical features
    ranges = {
        "Age": (14, 18),
        "Grade": (9, 12),
        "SES_Quartile": (1, 4),
        "AttendanceRate": (0.0, 1.0),
        "StudyHours": (0.0, 4.0),
        "InternetAccess": (0, 1),
        "Extracurricular": (0, 1),
        "PartTimeJob": (0, 1),
        "ParentSupport": (0, 1),
        "Romantic": (0, 1),
        "FreeTime": (1, 5),
        "GoOut": (1, 5),
        "GPA": (0.0, 4.0)
    }
    
    clipped_count = 0
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            before = len(df)
            # Clip values to range
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            after = len(df[(df[col] < min_val) | (df[col] > max_val)])
            if after < before:
                clipped_count += (df[col] < min_val).sum() + (df[col] > max_val).sum()
                if (df[col] < min_val).sum() > 0 or (df[col] > max_val).sum() > 0:
                    print(f"  Clipped {col} to range [{min_val}, {max_val}]")
    
    if clipped_count > 0:
        print(f"  [OK] Clipped {clipped_count:,} out-of-range values")
    
    return df

def validate_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Validate categorical values match expected categories.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        DataFrame with validated categorical values
    """
    df = df.copy()
    
    categorical_expected = {
        "ParentalEducation": ["<HS", "HS", "SomeCollege", "Bachelors+"],
        "SchoolType": ["Public", "Private"],
        "Locale": ["Suburban", "City", "Rural", "Town"]
    }
    
    for col, expected_values in categorical_expected.items():
        if col in df.columns:
            invalid = ~df[col].isin(expected_values)
            invalid_count = invalid.sum()
            if invalid_count > 0:
                print(f"  [WARNING] {invalid_count:,} invalid values in {col}")
                # Replace invalid values with mode or first valid value
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else expected_values[0]
                df.loc[invalid, col] = mode_val
                print(f"  Replaced invalid values in {col} with: {mode_val}")
    
    return df

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Detect and optionally handle outliers using IQR method.
    
    Args:
        df: DataFrame to check for outliers
        columns: List of columns to check (None = all numerical)
    
    Returns:
        DataFrame with outlier information (outliers are not removed by default)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    for col in columns:
        if col in df.columns and col != "pass_fail":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_pct = (outliers / len(df)) * 100
                outlier_info[col] = {
                    "count": outliers,
                    "percentage": outlier_pct,
                    "bounds": (lower_bound, upper_bound)
                }
    
    if outlier_info:
        print(f"\n  Outlier Detection (IQR method):")
        for col, info in outlier_info.items():
            print(f"    {col}: {info['count']:,} outliers ({info['percentage']:.2f}%)")
        print(f"  Note: Outliers are kept (may contain valuable information)")
    
    return df

def load_dataset(split: str = "train", input_data_dir: Path = None) -> pd.DataFrame:
    """Load and validate dataset split from input data directory.
    
    Args:
        split: One of 'train', 'validation', 'test'
        input_data_dir: Path to input data directory (default: data/raw)
    
    Returns:
        DataFrame with loaded and validated data
    """
    if input_data_dir is None:
        input_data_dir = Path("data/raw")
    else:
        input_data_dir = Path(input_data_dir)
    
    file_path = input_data_dir / f"{split}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading {split} dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Validate and clean data
    print(f"  Validating data types...")
    df = validate_data_types(df)
    
    print(f"  Validating feature ranges...")
    df = validate_feature_ranges(df)
    
    print(f"  Validating categorical values...")
    df = validate_categorical_values(df)
    
    print(f"  Handling missing values...")
    df = handle_missing_values(df, strategy="drop")
    
    print(f"  [OK] Data validation complete ({len(df):,} rows remaining)")
    
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary pass/fail target using GPA + test score strategy.
    
    Uses a clear labeling strategy:
    - Pass (1): BOTH conditions must be true - GPA >= 2.0 AND AvgTestScore >= 73
    - Fail (0): Remaining students (who don't meet all Pass conditions)
    - Attendance is no longer part of the label (still used as a predictive feature)
    - No borderline cases removed - all students are labeled
    
    Args:
        df: DataFrame with GPA and test score columns
    
    Returns:
        DataFrame with added 'pass_fail' column (all rows labeled)
        - 1 = Pass (Good Standing)
        - 0 = Fail (Academic Risk)
    """
    required_cols = ["GPA", "TestScore_Math", "TestScore_Reading", "TestScore_Science"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing for labeling: {missing_cols}")
    
    df = df.copy()
    initial_rows = len(df)
    
    # Compute average test score
    df["AvgTestScore"] = (
        df["TestScore_Math"] + 
        df["TestScore_Reading"] + 
        df["TestScore_Science"]
    ) / 3.0
    
    # Define Pass conditions (ALL must be true)
    pass_conditions = (
        (df["GPA"] >= 2.0) &
        (df["AvgTestScore"] >= 73)
    )
    
    # Label Pass (1) if all conditions are met, otherwise Fail (0)
    df["pass_fail"] = pass_conditions.astype(int)
    
    # Convert to int
    df["pass_fail"] = df["pass_fail"].astype(int)
    
    # Print detailed statistics (before dropping AvgTestScore)
    pass_count = (df["pass_fail"] == 1).sum()
    fail_count = (df["pass_fail"] == 0).sum()
    pass_rate = pass_count / len(df) if len(df) > 0 else 0
    
    print(f"\nLabeling Statistics:")
    print(f"  Total rows: {initial_rows:,}")
    print(f"  All rows labeled (no borderline cases removed)")
    print(f"\nTarget Distribution:")
    print(f"  Pass (1 - Good Standing): {pass_count:,} ({pass_rate*100:.2f}%)")
    print(f"  Fail (0 - Academic Crisis): {fail_count:,} ({(1-pass_rate)*100:.2f}%)")
    
    # Print condition breakdown for Pass cases
    if pass_count > 0:
        pass_df = df[df['pass_fail']==1]
        print(f"\nPass Condition Breakdown (all must be true):")
        print(f"  GPA >= 2.0: {(pass_df['GPA'] >= 2.0).sum():,}")
        if "AvgTestScore" in pass_df.columns:
            print(f"  AvgTestScore >= 73: {(pass_df['AvgTestScore'] >= 73).sum():,}")
    
    # Print why students failed (which conditions they missed)
    if fail_count > 0:
        fail_df = df[df['pass_fail']==0]
        print(f"\nFail Condition Analysis:")
        print(f"  Failed GPA condition (GPA < 2.0): {(fail_df['GPA'] < 2.0).sum():,}")
        if "AvgTestScore" in fail_df.columns:
            print(f"  Failed AvgTestScore condition (AvgTestScore < 73): {(fail_df['AvgTestScore'] < 73).sum():,}")
        
        # Count how many conditions students failed
        condition_failures = (
            (fail_df['GPA'] < 2.0).astype(int) +
            (fail_df['AvgTestScore'] < 73).astype(int)
        )
        print(f"\n  Number of conditions failed:")
        for i in range(1, 3):
            count = (condition_failures == i).sum()
            print(f"    {i} condition(s): {count:,} students")
    
    # Drop AvgTestScore (we don't want it in final features - test scores are excluded to prevent leakage)
    if "AvgTestScore" in df.columns:
        df = df.drop(columns=["AvgTestScore"])
    
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Prepare features by separating target and removing leakage features.
    
    Removes:
    - GPA (used to create target, would be direct leakage)
    - Test scores (TestScore_Math, TestScore_Reading, TestScore_Science): 
      Directly tied to GPA, would make prediction trivial
    
    Uses clean feature list focusing on behavioral and demographic factors.
    
    Args:
        df: DataFrame with features and target
    
    Returns:
        Tuple of (features_df, target_column_name)
    """
    target_col = "pass_fail"
    
    # Features to exclude (leakage)
    exclude_cols = [target_col, "GPA", "TestScore_Math", "TestScore_Reading", "TestScore_Science"]
    
    # Use clean feature list (only include features that exist in dataframe)
    feature_cols = [col for col in CLEAN_FEATURE_LIST if col in df.columns]
    
    # Also include any other valid columns not in exclude list (for safety)
    remaining_cols = [col for col in df.columns if col not in exclude_cols and col not in feature_cols]
    
    # Print excluded features for visibility
    excluded = [col for col in exclude_cols if col in df.columns and col != target_col]
    if excluded:
        print(f"\nExcluding leakage features: {excluded}")
    
    return df[feature_cols + remaining_cols], target_col

def create_probe_dataset(test_df: pd.DataFrame, n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a smaller probe dataset for benchmarking.
    
    Args:
        test_df: Full test dataset
        n_samples: Number of samples to include
        random_state: Random seed
    
    Returns:
        Probe dataset
    """
    n_samples = min(n_samples, len(test_df))
    probe_df = test_df.sample(n=n_samples, random_state=random_state).copy()
    print(f"\nCreated probe dataset with {len(probe_df):,} samples")
    return probe_df

def main():
    """Main data preparation pipeline."""
    parser = ArgumentParser(description="Prepare Student Performance Dataset")
    parser.add_argument(
        "--use-subsample",
        action="store_true",
        help="[DEPRECATED] This flag is deprecated. Standard split sizes are used (Train: 1M, Validation: 100K, Test: 300K)"
    )
    parser.add_argument(
        "--gpa-threshold",
        type=float,
        default=None,
        help="[DEPRECATED] GPA threshold is no longer used. Two-feature labeling strategy is used instead (Pass: GPA>=2.0 AND AvgTestScore>=73, Fail: All other students)"
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=5000,
        help="Number of samples for probe dataset (default: 5000)"
    )
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default="data/raw",
        help="Input directory path for raw data files (default: data/raw). For SageMaker, use /opt/ml/processing/input"
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default="data/processed",
        help="Output directory path for processed data files (default: data/processed). For SageMaker, use /opt/ml/processing/output"
    )
    args = parser.parse_args()
    
    # Convert to Path objects and ensure directories exist
    input_data_dir = Path(args.input_data_dir)
    output_data_dir = Path(args.output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Student Performance Dataset - Data Preparation")
    print("=" * 60)
    print(f"Using multi-condition labeling strategy:")
    print(f"  Pass (Good Standing): GPA >= 2.0 AND AvgTestScore >= 73")
    print(f"  Fail: All remaining students (who don't meet all Pass conditions)")
    print(f"  All students will be labeled (no borderline cases removed)")
    if args.gpa_threshold is not None:
        print(f"\n  [WARNING] --gpa-threshold parameter is deprecated and ignored")
    
    print(f"\nInput directory: {input_data_dir}")
    print(f"Output directory: {output_data_dir}")
    
    # Load datasets
    train_df = load_dataset("train", input_data_dir)
    validation_df = load_dataset("validation", input_data_dir)
    test_df = load_dataset("test", input_data_dir)
    
    # Apply split sizes: Train=1M, Validation=100K, Test=300K
    print("\n" + "=" * 60)
    print("Applying Dataset Split Sizes")
    print("=" * 60)
    print(f"Train: 1,000,000 rows")
    print(f"Validation: 100,000 rows")
    print(f"Test: 300,000 rows")
    print()
    
    # Limit to specified sizes
    train_df = train_df.head(1000000).copy()
    validation_df = validation_df.head(100000).copy()
    test_df = test_df.head(300000).copy()
    
    print(f"[OK] Train dataset: {len(train_df):,} rows")
    print(f"[OK] Validation dataset: {len(validation_df):,} rows")
    print(f"[OK] Test dataset: {len(test_df):,} rows")
    print()
    
    # Legacy subsample flag (now just prints a message)
    if args.use_subsample:
        print("Note: --use-subsample flag is deprecated. Using standard split sizes above.")
        print()
    
    # Create academic standing target using multi-condition strategy
    print("\n" + "=" * 60)
    print("Creating Pass/Fail labels using multi-condition strategy")
    print("=" * 60)
    
    print("\nProcessing Training Set:")
    train_df = create_target(train_df)
    
    print("\nProcessing Validation Set:")
    validation_df = create_target(validation_df)
    
    print("\nProcessing Test Set:")
    test_df = create_target(test_df)
    
    # Final data quality checks
    print("\n" + "=" * 60)
    print("Final Data Quality Checks")
    print("=" * 60)
    
    for name, df in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
        print(f"\n{name} Set:")
        
        # Missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            print(f"  [WARNING] Missing values: {missing:,} total")
            print(f"     Columns with missing: {list(missing_cols.index)}")
        else:
            print(f"  [OK] No missing values")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"  [WARNING] Duplicate rows: {duplicates:,}")
            df = df.drop_duplicates()
            print(f"     Removed duplicates: {len(df):,} rows remaining")
        else:
            print(f"  [OK] No duplicate rows")
        
        # Outlier detection (informational)
        detect_outliers_iqr(df)
        
        # Update dataframe
        if name == "Train":
            train_df = df
        elif name == "Validation":
            validation_df = df
        else:
            test_df = df
    
    # Save processed datasets
    print("\n" + "=" * 60)
    print("Saving processed datasets")
    print("=" * 60)
    
    # Save train/validation/test with target
    train_df.to_csv(output_data_dir / "train.csv", index=False)
    validation_df.to_csv(output_data_dir / "validation.csv", index=False)
    test_df.to_csv(output_data_dir / "test.csv", index=False)
    
    print(f"[OK] Saved {output_data_dir / 'train.csv'} ({len(train_df):,} rows)")
    print(f"[OK] Saved {output_data_dir / 'validation.csv'} ({len(validation_df):,} rows)")
    print(f"[OK] Saved {output_data_dir / 'test.csv'} ({len(test_df):,} rows)")
    
    # Create probe dataset for benchmarking
    probe_df = create_probe_dataset(test_df, n_samples=args.probe_samples)
    probe_df.to_csv(output_data_dir / "probe.csv", index=False)
    print(f"[OK] Saved {output_data_dir / 'probe.csv'} ({len(probe_df):,} rows)")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run EDA: python scripts/eda.py --split train")
    print(f"  2. Train model: python scripts/train.py")
    print(f"  3. Start API: python serve.py")

if __name__ == "__main__":
    main()

