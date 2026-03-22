FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    joblib \
    scikit-learn \
    xgboost \
    lightgbm \
    pandas \
    numpy \
    pymysql \
    sqlalchemy \
    cryptography \
    dill

COPY src/api/ ./src/api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]