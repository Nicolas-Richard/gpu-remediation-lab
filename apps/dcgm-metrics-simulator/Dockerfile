FROM python:3.12-alpine AS test

ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY app.py /app/app.py
COPY tests /app/tests
RUN python -m unittest discover -s /app/tests -v

FROM python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=test /app/app.py /app/app.py

USER 65532:65532
EXPOSE 9400
ENTRYPOINT ["python", "/app/app.py"]
