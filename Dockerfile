# Use an official Python runtime as a parent image
FROM python:3.11


WORKDIR /app

RUN pip install opencv-python
RUN pip install git+https://github.com/qubvel/segmentation_models
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
