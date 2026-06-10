FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV HOST=0.0.0.0
ENV PORT=8080

CMD ["python", "-m", "uvicorn", "relay_server:app", "--host", "0.0.0.0", "--port", "8080"]
