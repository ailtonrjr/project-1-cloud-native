FROM python:3.12-slim

WORKDIR /app

COPY task1_analysis/ ./task1_analysis/
COPY data/ ./data/

RUN pip install --no-cache-dir pandas numpy matplotlib seaborn

CMD ["python", "task1_analysis/data_analysis.py"]

