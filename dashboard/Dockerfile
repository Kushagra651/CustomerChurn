FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    requests \
    joblib \
    scikit-learn \
    xgboost \
    lightgbm

COPY dashboard/ ./dashboard/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]