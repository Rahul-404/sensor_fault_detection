FROM python:3.8.20-slim

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]