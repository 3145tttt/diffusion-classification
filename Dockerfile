FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements_predict.yml ./
RUN pip install -r requirements_predict.yml

ENTRYPOINT ["python", "predict.py"]