FROM ubuntu:22.04
ARG TENSORFLOW_VERSION="2.15"
RUN apt update
RUN apt install -y python3.11
RUN apt install -y python3-pip
RUN apt install -y git
RUN mkdir /root/.ssh/
COPY ./devops_creds_and_keys/service_ssh_keys/id_rsa /root/.ssh/id_rsa
COPY ./devops_creds_and_keys/service_ssh_keys/id_rsa.pub /root/.ssh/id_rsa.pub
RUN chmod 600 /root/.ssh/id_rsa
COPY ./devops_creds_and_keys/github-temp-key .
RUN cat github-temp-key >> ~/.ssh/known_hosts

RUN mkdir /app
WORKDIR /app
RUN git clone git@github.com:mylonasc/tf_gnns
RUN pip install tensorflow==${TENSORFLOW_VERSION}
RUN python3 -c "tf_to_tfp_dict = {'2.15':'0.22','2.16' : '0.23','2.17' : '0.24'} ; print(tf_to_tfp_dict['${TENSORFLOW_VERSION}'])" > /tmp/TFP_VERSION
RUN pip install tensorflow_probability==$(cat /tmp/TFP_VERSION)
RUN pip install tf_keras==${TENSORFLOW_VERSION}
RUN cat tf_gnns/requirements.txt | grep -v "tensorflow>=" | grep -v "tensorflow_prob" | grep -v 'tf_keras' > reqs.tmp.txt
RUN pip install -r reqs.tmp.txt
WORKDIR /app/tf_gnns
RUN python3 test.py
#RUN cd tf_gnns && python3 test.py

ENTRYPOINT ["bash"] 
