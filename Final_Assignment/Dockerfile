FROM python:3.7.9

WORKDIR /app

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
 	build-essential \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get dist-upgrade
RUN apt-get update --yes
RUN apt-get dist-upgrade
RUN /usr/local/bin/python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY ./container_files .
RUN mkdir -p 'feature plots' 'ranking plots' 'msd plots' 'brute force plots'
RUN mkdir -p ../output

RUN pip install biopython

RUN python import_graph_train_test.py
RUN python predict.py

RUN rm flexibility_secondary_etc.csv
RUN rm import_graph_train_test.py
RUN rm predict.py

CMD mv * ../output