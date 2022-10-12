FROM python:3.7
ADD . /src
RUN pip install kopf
RUN pip install kubernetes certvalidator pyngrok
CMD kopf run /src/handlers.py --verbose
