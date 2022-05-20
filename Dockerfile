# syntax=docker/dockerfile:1
FROM gcr.io/gnomondigital-sandbox/tvmbase
ENV APP_HOME /app
WORKDIR ${APP_HOME}
RUN mkdir outputs/
RUN sar -u 10 > outputs/sar_u.txt & sar -r 10 > outputs/sar_r.txt &
COPY . /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python main.py --path example1.json > outputs/execution_log.txt
