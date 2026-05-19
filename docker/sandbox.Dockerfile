FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends git ripgrep \
  && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/sh sandbox
WORKDIR /workspace

COPY rlm_harness/sandbox/worker.py /opt/rlm-harness/worker.py
COPY rlm_harness/sandbox/rlm_shim.py /opt/rlm-harness/rlm_shim.py
COPY rlm_harness/sandbox/tools.py /opt/rlm-harness/sandbox_tools.py
RUN chmod 0555 /opt/rlm-harness/worker.py /opt/rlm-harness/rlm_shim.py \
  /opt/rlm-harness/sandbox_tools.py

USER sandbox
CMD ["python", "/opt/rlm-harness/worker.py"]
