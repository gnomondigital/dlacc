# syntax=docker/dockerfile:1
FROM ubuntu:18.04
COPY . /app
CMD nohup python3.9 /app/main.py --config example0.json > outputs/execution_log.txt
