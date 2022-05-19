# syntax=docker/dockerfile:1
FROM tvmbase
COPY . /app
ENV APP_HOME /app
WORKDIR ${APP_HOME}
# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir outputs/
RUN sar -u 10 > outputs/sar_u.txt & sar -r 10 > outputs/sar_r.txt &
CMD python main.py --path example1.json > outputs/execution_log.txt
