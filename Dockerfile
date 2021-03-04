FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api /api
COPY game_one /game_one
COPY content_base_svd.pickle /content_base_svd.pickle
COPY knn_model.pickle /knn_model.pickle
COPY model-cfm.pickle /model-cfm.pickle
COPY preproc.pickle /preproc.pickle


CMD uvicorn api.fast:app --host 0.0.0.0 --port 8080
