FROM python:3.8-slim-buster

COPY . /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 9000

CMD ["python3", "/app/src/exporter.py"]