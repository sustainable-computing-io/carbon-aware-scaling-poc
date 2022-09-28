FROM python:3.8-slim-buster

COPY /app /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 9000

RUN ls -la /app/src/handlers

CMD ["python3", "/app/src/handlers/co2signal_collector.py"]