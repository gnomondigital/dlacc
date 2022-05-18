# syntax=docker/dockerfile:1
FROM octoml/tvm
COPY . /app
ENV APP_HOME /app
WORKDIR ${APP_HOME}
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir outputs/
RUN sar -u 10 > outputs/sar_u.txt & sar -r 10 > outputs/sar_r.txt &
CMD python main.py --path example1.json > outputs/execution_log.txt
