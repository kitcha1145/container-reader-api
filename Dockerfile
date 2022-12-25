FROM aiport/alpr-env:1.0.3

RUN mkdir /root/.ssh/

ADD id_rsa /root/.ssh/id_rsa

RUN touch /root/.ssh/known_hosts

RUN apt-get install unzip -y

RUN ssh-keyscan gitlab.com >> /root/.ssh/known_hosts

WORKDIR /workspace/

RUN git clone git@gitlab.com:ai-team4/container_api_op.git

WORKDIR container_api_op/aiApi/apps/ml/detection/container_reader

COPY ./p.txt .

RUN sh get_m.sh

WORKDIR /workspace/container_api_op/aiApi

RUN mkdir -p templates

RUN mkdir -p media

RUN chmod +x ep.sh

ENTRYPOINT ./ep.sh