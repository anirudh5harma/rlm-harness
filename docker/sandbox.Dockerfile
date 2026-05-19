FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/sh sandbox
WORKDIR /workspace

COPY rlm_harness/sandbox/worker.py /opt/rlm-harness/worker.py
COPY rlm_harness/sandbox/rlm_shim.py /opt/rlm-harness/rlm_shim.py
RUN chmod 0555 /opt/rlm-harness/worker.py /opt/rlm-harness/rlm_shim.py

USER sandbox
CMD ["python", "/opt/rlm-harness/worker.py"]
