"""
churn_etl_dag.py
-----------------
Airflow DAG for the PulseDirects Customer Churn ETL Pipeline.

Schedule : Monthly (1st of every month at 00:00)
Tasks    :
  1. validate_data_source  — check MySQL connection + row counts
  2. run_feature_engineering — extract, clean, engineer, save preprocessed CSV
  3. run_model_training      — train 7 models, save best as churn_model.pkl
  4. run_model_evaluation    — generate evaluation plots + report JSON
  5. log_pipeline_summary    — log final metrics to Airflow logs

All tasks run sequentially. If any task fails, downstream
tasks are skipped and the failure is logged.
"""

import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ── Default args ───────────────────────────────────────────────────────────────
default_args = {
    'owner'            : 'churn_pipeline',
    'depends_on_past'  : False,
    'start_date'       : datetime(2024, 1, 1),
    'retries'          : 1,
    'retry_delay'      : timedelta(minutes=5),
    'email_on_failure' : False,   # using log-based alerting instead
    'email_on_retry'   : False,
}

# ── DAG definition ─────────────────────────────────────────────────────────────
dag = DAG(
    dag_id='churn_etl_pipeline',
    default_args=default_args,
    description='Monthly ETL pipeline for PulseDirects churn prediction',
    schedule_interval='0 0 1 * *',   # 1st of every month at midnight
    catchup=False,
    max_active_runs=1,
    tags=['churn', 'etl', 'ml', 'pulsedirects'],
)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — VALIDATE DATA SOURCE
# ═════════════════════════════════════════════════════════════════════════════

def validate_data_source(**context):
    """
    Checks MySQL connection and validates minimum row counts
    before starting the pipeline. Fails fast if data is missing
    so downstream tasks don't run on empty data.
    """
    import os
    import sqlalchemy

    log.info("=" * 55)
    log.info("TASK 1: Validating data source")
    log.info("=" * 55)

    host     = os.getenv("CHURN_DB_HOST",     "mysql")
    port     = os.getenv("CHURN_DB_PORT",     "3306")
    db       = os.getenv("CHURN_DB_NAME",     "pulsedirects")
    user     = os.getenv("CHURN_DB_USER",     "root")
    password = os.getenv("CHURN_DB_PASSWORD", "root")

    engine = sqlalchemy.create_engine(
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    )

    MIN_ROWS = {
        'customers'      : 1000,
        'payments'       : 1000,
        'support_tickets': 1000,
    }

    with engine.connect() as conn:
        for table, min_count in MIN_ROWS.items():
            result = conn.execute(
                sqlalchemy.text(f"SELECT COUNT(*) FROM {table}")
            )
            count = result.scalar()
            log.info(f"  {table}: {count} rows")

            if count < min_count:
                raise ValueError(
                    f"Table '{table}' has only {count} rows — "
                    f"minimum required is {min_count}. Aborting pipeline."
                )

    log.info("Data source validation passed.")

    # Push row counts to XCom for summary task
    context['ti'].xcom_push(key='validation_status', value='passed')
    context['ti'].xcom_push(key='run_date', value=str(datetime.now().date()))


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def run_feature_engineering(**context):
    """
    Imports and runs the feature engineering pipeline.
    Reads from MySQL, engineers features, saves preprocessed CSV.
    """
    import sys
    import os
    sys.path.append('/opt/airflow/src')

    log.info("=" * 55)
    log.info("TASK 2: Running feature engineering")
    log.info("=" * 55)

    from data_preprocessing.feature_engineering import run_feature_engineering
    run_feature_engineering()

    log.info("Feature engineering complete.")
    context['ti'].xcom_push(key='feature_engineering_status', value='success')


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def run_model_training(**context):
    """
    Imports and runs model training pipeline.
    Trains 7 models with GridSearchCV, saves best model as .pkl.
    """
    import sys
    sys.path.append('/opt/airflow/src')

    log.info("=" * 55)
    log.info("TASK 3: Running model training")
    log.info("=" * 55)

    from models.train_model import run_training
    run_training()

    # Read best model info and push to XCom
    import json
    from pathlib import Path
    info_path = Path('/opt/airflow/models/best_model_info.json')

    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        context['ti'].xcom_push(key='best_model',   value=info['best_model'])
        context['ti'].xcom_push(key='roc_auc',      value=info['roc_auc'])
        context['ti'].xcom_push(key='accuracy',     value=info['accuracy'])
        context['ti'].xcom_push(key='training_status', value='success')
        log.info(f"Best model: {info['best_model']} | ROC-AUC: {info['roc_auc']}")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_model_evaluation(**context):
    """
    Imports and runs model evaluation.
    Generates comparison chart, PR curve, and evaluation report JSON.
    """
    import sys
    sys.path.append('/opt/airflow/src')

    log.info("=" * 55)
    log.info("TASK 4: Running model evaluation")
    log.info("=" * 55)

    from models.evaluate_model import run_evaluation
    run_evaluation()

    context['ti'].xcom_push(key='evaluation_status', value='success')
    log.info("Evaluation complete. Plots and report saved.")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 5 — LOG PIPELINE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def log_pipeline_summary(**context):
    """
    Pulls results from XCom and logs a full pipeline summary.
    This is the single source of truth for each monthly run.
    Acts as both success and failure summary log.
    """
    ti = context['ti']

    validation = ti.xcom_pull(task_ids='validate_data_source',    key='validation_status')
    fe_status  = ti.xcom_pull(task_ids='run_feature_engineering',  key='feature_engineering_status')
    tr_status  = ti.xcom_pull(task_ids='run_model_training',       key='training_status')
    ev_status  = ti.xcom_pull(task_ids='run_model_evaluation',     key='evaluation_status')
    best_model = ti.xcom_pull(task_ids='run_model_training',       key='best_model')
    roc_auc    = ti.xcom_pull(task_ids='run_model_training',       key='roc_auc')
    accuracy   = ti.xcom_pull(task_ids='run_model_training',       key='accuracy')
    run_date   = ti.xcom_pull(task_ids='validate_data_source',     key='run_date')

    log.info("=" * 55)
    log.info("PIPELINE RUN SUMMARY")
    log.info("=" * 55)
    log.info(f"  Run date             : {run_date}")
    log.info(f"  Data validation      : {validation}")
    log.info(f"  Feature engineering  : {fe_status}")
    log.info(f"  Model training       : {tr_status}")
    log.info(f"  Model evaluation     : {ev_status}")
    log.info(f"  Best model           : {best_model}")
    log.info(f"  ROC-AUC              : {roc_auc}")
    log.info(f"  Accuracy             : {accuracy}")
    log.info("=" * 55)

    # Determine overall status
    all_statuses = [validation, fe_status, tr_status, ev_status]
    if all(s == 'success' or s == 'passed' for s in all_statuses if s):
        log.info("MONTHLY RUN STATUS: SUCCESS")
    else:
        log.warning("MONTHLY RUN STATUS: PARTIAL FAILURE — check individual task logs")


# ═════════════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

with dag:

    start = EmptyOperator(task_id='start')

    task_validate = PythonOperator(
        task_id='validate_data_source',
        python_callable=validate_data_source,
        provide_context=True,
    )

    task_feature_eng = PythonOperator(
        task_id='run_feature_engineering',
        python_callable=run_feature_engineering,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )

    task_training = PythonOperator(
        task_id='run_model_training',
        python_callable=run_model_training,
        provide_context=True,
        execution_timeout=timedelta(minutes=60),  # tuning takes time
    )

    task_evaluation = PythonOperator(
        task_id='run_model_evaluation',
        python_callable=run_model_evaluation,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
    )

    task_summary = PythonOperator(
        task_id='log_pipeline_summary',
        python_callable=log_pipeline_summary,
        provide_context=True,
        trigger_rule='all_done',  # runs even if upstream tasks fail
    )

    end = EmptyOperator(task_id='end')

    # ── Pipeline order ─────────────────────────────────────────────
    start >> task_validate >> task_feature_eng >> task_training >> task_evaluation >> task_summary >> end