import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)
 

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH   = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"
PREPROC_DIR     = BASE_DIR / "data" / "preprocessed"
OUTPUT_PATH     = PREPROC_DIR / "churn_preprocessed.csv"
FEAT_COLS_PATH  = PREPROC_DIR / "feature_columns.json"
 

COLUMNS_TO_DROP = [
    'CustomerID', 'Count', 'Country', 'State',
    'Lat Long', 'Latitude', 'Longitude',
    'Zip Code', 'City',           # City: too many unique values
    'Churn Reason', 'Churn Label',
    'Churn Score', 'CLTV',
    'Customer Id',
    'Created At',
    'Updated At'         # leakage — derived from churn outcome
]

BINARY_COLS = [
    'Partner', 'Dependents', 'Phone Service', 'Paperless Billing',
    'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming Tv', 'Streaming Movies'
]
 

CATEGORICAL_COLS = ['Contract', 'Internet Service', 'Payment Method', 'Multiple Lines'] 
NUMERICAL_COLS = ['Tenure Months', 'Monthly Charges', 'Total Charges']
 
 
# load data from sql
def load_from_mysql() -> pd.DataFrame:
    try:
        import sqlalchemy
        host     = os.getenv("CHURN_DB_HOST", "localhost")
        port     = os.getenv("CHURN_DB_PORT", "3307")
        db       = os.getenv("CHURN_DB_NAME", "pulsedirects")
        user     = os.getenv("CHURN_DB_USER", "root")
        password = os.getenv("CHURN_DB_PASSWORD", "root")
 
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        )
 
        query = """
            SELECT
                c.*,
                p.monthly_Charges   AS `monthly Charges`,
                p.total_Charges     AS `total Charges`,
                p.payment_Method    AS `payment Method`,
                p.paperless_Billing AS `paperless Billing`,
                COUNT(st.ticket_id) AS support_ticket_count
            FROM customers c
            LEFT JOIN payments p
                ON c.customer_id = p.customer_id
            LEFT JOIN support_tickets st
                ON c.customer_id = st.customer_id
            GROUP BY c.customer_id, p.monthly_Charges, p.total_Charges,
                     p.payment_Method, p.paperless_Billing
        """
 
        df = pd.read_sql(query, engine)
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        log.info(f"Loaded {len(df)} rows from MySQL.")
        return df
 
    except Exception as e:
        log.warning(f"MySQL load failed: {e}")
        raise
 
 
def load_from_excel() -> pd.DataFrame:
    """Fallback: reads directly from raw Excel file."""
    log.info(f"Loading from Excel: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH)
    log.info(f"Loaded {len(df)} rows from Excel.")
    return df
 
 
def load_data() -> pd.DataFrame:
    """Try MySQL first, fall back to Excel."""
    try:
        return load_from_mysql()
    except Exception:
        log.warning("Falling back to Excel file.")
        return load_from_excel()
 
 
# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEANING
# ═════════════════════════════════════════════════════════════════════════════
 
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier, leakage, and high-cardinality columns."""
    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing)
    log.info(f"Dropped {len(existing)} columns.")
    return df
 
 
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total Charges sometimes loads as object (blank spaces in raw data).
    Force numeric, let NaNs be handled in imputation step.
    """
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    return df
 
 
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values before feature engineering.
    Numerical  → median  (robust to outliers)
    Categorical → mode   (most frequent value)
    """
    # Numerical
    for col in ['Total Charges', 'Monthly Charges', 'Tenure Months']:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log.info(f"Imputed '{col}' with median={median_val:.2f}")
 
    # Categorical
    for col in BINARY_COLS + CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            log.info(f"Imputed '{col}' with mode='{mode_val}'")
 
    # Target
    if 'Churn Value' in df.columns and df['Churn Value'].isnull().any():
        df['Churn Value'].fillna(df['Churn Value'].median(), inplace=True)
 
    return df
 
 
# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from existing columns.
 
    Features added:
    ─────────────────────────────────────────────────────
    average_charges     : Total Charges / Tenure Months
                          How much the customer pays on average per month.
                          More stable than Monthly Charges alone since it
                          accounts for historical billing changes.
 
    high_value_customer : 1 if Monthly Charges > 75th percentile threshold
                          Flags premium customers — high value customers
                          who churn are a bigger business loss.
    ─────────────────────────────────────────────────────
    """
    df = df.copy()
 
    # Average Charges — guard against division by zero (new customers)
    df['average_charges'] = np.where(
        df['Tenure Months'] > 0,
        df['Total Charges'] / df['Tenure Months'],
        df['Monthly Charges']   # if tenure=0, use monthly as proxy
    )
    df['average_charges'] = df['average_charges'].round(2)
    log.info("Engineered: average_charges")
 
    # High value customer flag
    threshold = df['Monthly Charges'].quantile(0.75)
    df['high_value_customer'] = (df['Monthly Charges'] > threshold).astype(int)
    log.info(f"Engineered: high_value_customer (threshold={threshold:.2f})")
 
    return df
 
 
# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — ENCODING
# ═════════════════════════════════════════════════════════════════════════════
 
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes all categorical columns to numeric.
 
    Strategy:
    ─────────────────────────────────────────────────────
    Binary yes/no columns  → map to 1/0 directly
                             'No internet service' / 'No phone service'
                             treated as 0 (feature not available)
 
    Gender                 → Female=0, Male=1
 
    Senior Citizen         → already 0/1, skip
 
    Multi-class columns    → pandas category codes
                             consistent mapping saved to JSON
                             so FastAPI can apply same encoding
    ─────────────────────────────────────────────────────
    """
    df = df.copy()
 
    # Binary columns
    binary_map = {
        'Yes': 1, 'No': 0,
        'No phone service': 0,
        'No internet service': 0
    }
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
 
    # Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
 
    # Multi-class categoricals → integer codes
    encoding_map = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
            encoding_map[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes
 
    log.info(f"Encoded binary columns: {BINARY_COLS}")
    log.info(f"Encoded categorical columns: {CATEGORICAL_COLS}")
 
    return df, encoding_map
 
 
# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE
# ═════════════════════════════════════════════════════════════════════════════
 
def save_outputs(df: pd.DataFrame, encoding_map: dict) -> None:
    """Save preprocessed CSV and metadata."""
    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
 
    # Preprocessed dataset
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved preprocessed data → {OUTPUT_PATH}")
 
    # Feature columns list (used by train_model.py)
    feature_cols = [c for c in df.columns if c != 'Churn Value']
    metadata = {
        "feature_columns": feature_cols,
        "target_column": "Churn Value",
        "encoding_map": encoding_map,
        "numerical_cols": NUMERICAL_COLS + ['average_charges'],
        "engineered_cols": ['average_charges', 'high_value_customer'],
        "n_rows": len(df),
        "n_features": len(feature_cols)
    }
 
    with open(FEAT_COLS_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved feature metadata → {FEAT_COLS_PATH}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
 
def run_feature_engineering() -> None:
    log.info("=" * 55)
    log.info("START: Feature Engineering Pipeline")
    log.info("=" * 55)
 
    # 1. Load
    df = load_data()
    log.info(f"Raw shape: {df.shape}")
 
    # 2. Clean
    df = drop_columns(df)
    df = fix_dtypes(df)
    df = impute_missing(df)
    log.info(f"After cleaning shape: {df.shape}")
 
    # 3. Engineer
    df = engineer_features(df)
 
    # 4. Encode
    df, encoding_map = encode_features(df)
    log.info(f"Final shape: {df.shape}")
 
    # 5. Validate — no nulls should remain
    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning(f"Nulls remaining after preprocessing:\n{null_counts[null_counts > 0]}")
    else:
        log.info("Null check passed — no missing values.")
 
    # 6. Save
    save_outputs(df, encoding_map)
 
    log.info("=" * 55)
    log.info("DONE: Feature Engineering Pipeline")
    log.info("=" * 55)
 
 
if __name__ == "__main__":
    run_feature_engineering()