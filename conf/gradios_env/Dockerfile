FROM python:3.10

COPY requirements.txt /opt/app/requirements.txt
COPY model.py /opt/app/model.py
WORKDIR /opt/app

RUN pip install -r requirements.txt

EXPOSE 7860

CMD python3 model.py

# Social determinants of health and diabetes: A scientific review