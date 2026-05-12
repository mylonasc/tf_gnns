FROM python:3.11-slim

ARG TENSORFLOW_VERSION="2.15"

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

RUN chmod +x scripts/run_tf_matrix_tests.sh
RUN scripts/run_tf_matrix_tests.sh "${TENSORFLOW_VERSION}"

ENTRYPOINT ["bash"]
