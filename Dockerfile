FROM ubuntu:22.04
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

ARG TENSORFLOW_VERSION="2.15"
RUN pip install tensorflow==${TENSORFLOW_VERSION}

RUN python3 -c "tf_to_tfp_dict = {'2.10' : '0.20', '2.13' : '0.21','2.14' : '0.22','2.15':'0.22','2.16' : '0.23','2.17' : '0.24'} ; print(tf_to_tfp_dict['${TENSORFLOW_VERSION}'[:4]])" > /tmp/TFP_VERSION
RUN pip install tensorflow_probability==$(cat /tmp/TFP_VERSION)

## Numpy 1.x for tensorflow before 2.14
RUN python3 -c "TF_VERSION='${TENSORFLOW_VERSION}'; vers_splitted = TF_VERSION.split('.'); print(int(vers_splitted[1]) <= 14)" > /tmp/NEEDS_NUMPY_1X
RUN if [ $(cat /tmp/NEEDS_NUMPY_1X)==True ] ; then pip install numpy==1.26 ; fi

RUN python3 -c "import tensorflow as tf ; major, minor , _ = [int(i) for i in tf.__version__.split('.')]; print(minor >=16, ' ', '.'.join([str(major), str(minor)]))">>/tmp/NEEDS_KERAS2
RUN if [ $(cat /tmp/NEEDS_KERAS2 | awk '{print 1}') == "True" ] ; then pip install tf_keras==$(cat /tmp/NEEDS_KERAS2| awk '{print $2}'); fi
RUN cat tf_gnns/requirements.txt | grep -v "tensorflow>=" | grep -v "tensorflow_prob" | grep -v 'tf_keras' > reqs.tmp.txt

RUN pip install -r reqs.tmp.txt
WORKDIR /app/tf_gnns
RUN python3 test.py

ENTRYPOINT ["bash"] 
