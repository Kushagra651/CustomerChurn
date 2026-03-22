FROM apache/airflow:2.8.1
USER root
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir \
    pymysql \
    sqlalchemy \
    cryptography \
    joblib \
    scikit-learn \
    xgboost \
    lightgbm \
    pandas \
    numpy \
    matplotlib \
    seaborn