FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install pandas matplotlib scikit-learn joblib

CMD ["python", "hello_world_my.py"]