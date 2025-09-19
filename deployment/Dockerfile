FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src /app/src
COPY models /app/models
COPY deployment/api.py /app/api.py
CMD ["python", "api.py"]