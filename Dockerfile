# Reproducible CPU environment for Q-EpiGNN:  docker build -t qepignn . && docker run --rm qepignn
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
CMD ["pytest", "-q"]
