"""
seed_data.py
-------------
One-time script to load raw Excel data into MySQL.
Run this ONCE after schema.sql has been executed.

Order:
  1. docker-compose up -d mysql
  2. Apply schema.sql  (already done)
  3. python database/seed_data.py
  4. Verify with: SELECT COUNT(*) FROM customers;

Tables populated:
  - customers       (from Excel — one row per customer)
  - payments        (from Excel — billing columns)
  - support_tickets (simulated — random but realistic data)
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import sqlalchemy
from pathlib import Path
from datetime import datetime, timedelta

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"

# ── DB config ─────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("CHURN_DB_HOST",     "localhost")
DB_PORT     = os.getenv("CHURN_DB_PORT",     "3307")
DB_NAME     = os.getenv("CHURN_DB_NAME",     "pulsedirects")
DB_USER     = os.getenv("CHURN_DB_USER",     "root")
DB_PASSWORD = os.getenv("CHURN_DB_PASSWORD", "root")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = sqlalchemy.create_engine(url, echo=False)
    log.info("Database engine created.")
    return engine


def load_excel() -> pd.DataFrame:
    log.info(f"Reading Excel: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# BUILD CUSTOMERS TABLE
# ═════════════════════════════════════════════════════════════════════════════

def build_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Map Excel columns → customers table schema."""
    customers = pd.DataFrame()

    customers['customer_id']        = df['CustomerID'].astype(str).str.strip()
    customers['city']               = df['City'].astype(str).str.strip()
    customers['zip_code']           = pd.to_numeric(df['Zip Code'], errors='coerce')
    customers['latitude']           = pd.to_numeric(df['Latitude'], errors='coerce')
    customers['longitude']          = pd.to_numeric(df['Longitude'], errors='coerce')
    customers['gender']             = df['Gender'].astype(str).str.strip()
    customers['senior_citizen']     = pd.to_numeric(df['Senior Citizen'], errors='coerce').fillna(0).astype(int)
    customers['partner']            = df['Partner'].astype(str).str.strip()
    customers['dependents']         = df['Dependents'].astype(str).str.strip()
    customers['tenure_months']      = pd.to_numeric(df['Tenure Months'], errors='coerce').fillna(0).astype(int)
    customers['phone_service']      = df['Phone Service'].astype(str).str.strip()
    customers['multiple_lines']     = df['Multiple Lines'].astype(str).str.strip()
    customers['internet_service']   = df['Internet Service'].astype(str).str.strip()
    customers['online_security']    = df['Online Security'].astype(str).str.strip()
    customers['online_backup']      = df['Online Backup'].astype(str).str.strip()
    customers['device_protection']  = df['Device Protection'].astype(str).str.strip()
    customers['tech_support']       = df['Tech Support'].astype(str).str.strip()
    customers['streaming_tv']       = df['Streaming TV'].astype(str).str.strip()
    customers['streaming_movies']   = df['Streaming Movies'].astype(str).str.strip()
    customers['contract']           = df['Contract'].astype(str).str.strip()
    customers['churn_value']        = pd.to_numeric(df['Churn Value'], errors='coerce').fillna(0).astype(int)

    log.info(f"Customers built: {len(customers)} rows.")
    return customers


# ═════════════════════════════════════════════════════════════════════════════
# BUILD PAYMENTS TABLE
# ═════════════════════════════════════════════════════════════════════════════

def build_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Map billing columns → payments table schema."""
    payments = pd.DataFrame()

    payments['customer_id']       = df['CustomerID'].astype(str).str.strip()
    payments['paperless_billing'] = df['Paperless Billing'].astype(str).str.strip()
    payments['payment_method']    = df['Payment Method'].astype(str).str.strip()
    payments['monthly_charges']   = pd.to_numeric(df['Monthly Charges'], errors='coerce')
    payments['total_charges']     = pd.to_numeric(df['Total Charges'],   errors='coerce')

    # Fill nulls with median before inserting
    payments['monthly_charges'].fillna(payments['monthly_charges'].median(), inplace=True)
    payments['total_charges'].fillna(payments['total_charges'].median(),     inplace=True)

    log.info(f"Payments built: {len(payments)} rows.")
    return payments


# ═════════════════════════════════════════════════════════════════════════════
# BUILD SUPPORT TICKETS TABLE (simulated)
# ═════════════════════════════════════════════════════════════════════════════

def build_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate support ticket data since the Excel has none.

    Logic (realistic):
    - Churned customers (churn_value=1) get MORE tickets on average (3-6)
    - Non-churned customers get fewer tickets (0-3)
    - Severity and resolution time also skew worse for churned customers
    - Not every customer has a ticket
    """
    random.seed(42)
    np.random.seed(42)

    issue_types = [
        'Billing Issue', 'Network Outage', 'Slow Internet',
        'Service Cancellation', 'Equipment Fault', 'Account Access'
    ]
    severities = ['Low', 'Medium', 'High']
    statuses   = ['Resolved', 'Pending', 'Escalated']

    records   = []
    base_date = datetime(2023, 1, 1)

    for _, row in df.iterrows():
        cid        = str(row['CustomerID']).strip()
        is_churned = int(row.get('Churn Value', 0))
        tenure     = int(row.get('Tenure Months', 1))

        # Churned customers had more issues
        n_tickets = np.random.randint(3, 7) if is_churned else np.random.randint(0, 4)

        for _ in range(n_tickets):
            severity = (
                random.choices(severities, weights=[20, 40, 40])[0]
                if is_churned
                else random.choices(severities, weights=[50, 35, 15])[0]
            )
            status = (
                random.choices(statuses, weights=[50, 25, 25])[0]
                if is_churned
                else random.choices(statuses, weights=[80, 15, 5])[0]
            )
            days_offset     = random.randint(0, max(tenure * 30, 1))
            resolution_days = (
                random.randint(3, 14) if severity == 'High'
                else random.randint(1, 7)
            )

            records.append({
                'customer_id':      cid,
                'issue_type':       random.choice(issue_types),
                'severity':         severity,
                'resolution_days':  resolution_days,
                'status':           status,
                'created_at':       base_date + timedelta(days=days_offset)
            })

    tickets = pd.DataFrame(records)
    log.info(f"Support tickets simulated: {len(tickets)} rows.")
    return tickets


# ═════════════════════════════════════════════════════════════════════════════
# INSERT INTO MYSQL
# ═════════════════════════════════════════════════════════════════════════════

def insert_table(df: pd.DataFrame, table: str, engine) -> None:
    """Insert dataframe into MySQL table in chunks."""
    df.to_sql(
        name=table,
        con=engine,
        if_exists='append',     # schema already created by schema.sql
        index=False,
        chunksize=500,
        method='multi'
    )
    log.info(f"Inserted {len(df)} rows into '{table}'.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def seed():
    log.info("=" * 55)
    log.info("START: Database Seeding")
    log.info("=" * 55)

    engine = get_engine()
    df     = load_excel()

    # Build each table
    customers = build_customers(df)
    payments  = build_payments(df)
    tickets   = build_support_tickets(df)

    # Insert in FK order — customers first
    log.info("Inserting customers...")
    insert_table(customers, 'customers', engine)

    log.info("Inserting payments...")
    insert_table(payments, 'payments', engine)

    log.info("Inserting support_tickets...")
    insert_table(tickets, 'support_tickets', engine)

    # Verify row counts
    with engine.connect() as conn:
        for table in ['customers', 'payments', 'support_tickets']:
            result = conn.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table}"))
            count  = result.scalar()
            log.info(f"  {table}: {count} rows")

    log.info("=" * 55)
    log.info("DONE: Database Seeding Complete")
    log.info("=" * 55)


if __name__ == "__main__":
    seed()