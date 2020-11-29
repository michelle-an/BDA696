FROM ubuntu:focal

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
 	build-essential \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get dist-upgrade
RUN apt-get update --yes
RUN apt-get install --yes mysql-client
RUN apt-get install --yes mysql-server
RUN apt-get dist-upgrade

COPY run.sh run.sh
COPY ./baseball.sql baseball.sql
RUN mkdir $APP_HOME/output
RUN ["chmod", "u+x", "run.sh"]

SHELL ["/bin/bash", "-c"] 
CMD ./run.sh

# keep container alive by running something in the foreground (dev)
# CMD tail -f /dev/null
