FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# Pinned Python dependencies (squidpy pulls in spatialdata, dask and friends). Wheels only, except
# docrep (a squidpy dependency, pure Python, Apache-2.0) which is published as an sdist only.
COPY requirements.txt /processor/requirements.txt
RUN pip install --only-binary=:all: --no-binary=docrep -r /processor/requirements.txt

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# Build matplotlib's font cache now (as the run-time user would), so a run needs no writes outside /tmp.
# Every cache dir that exists at build time is made writable for uid 1000 (numba compiles at run time).
RUN python -c "import matplotlib.pyplot" && mkdir -p /tmp/numba && chmod -R a+rwX /tmp/matplotlib /tmp/numba

# Runs as uid 1000 on the platform; nothing here needs root.
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
