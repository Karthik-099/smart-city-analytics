# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Update pip and install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project files
COPY . .

# Expose ports for Streamlit (8501), Flask API (5000), MLflow UI (5000)
EXPOSE 8501 5000

# Default command: Run Streamlit dashboard
CMD ["streamlit", "run", "deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0"]