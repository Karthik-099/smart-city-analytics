
FROM python:3.10-slim AS builder

WORKDIR /app


RUN pip install --upgrade pip
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt


FROM python:3.10-slim

WORKDIR /app


COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8501 5000


CMD ["streamlit", "run", "deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0"]