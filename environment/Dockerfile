FROM --platform=linux/amd64 python:3.12.15-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends libquadmath0=14.2.0-19 libstdc++6=14.2.0-19 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.runtime.lock.txt /tmp/requirements.runtime.lock.txt
RUN python -m pip install --no-cache-dir --timeout 120 --retries 5 --require-hashes --only-binary=:all: --no-deps -r /tmp/requirements.runtime.lock.txt \
    && python -m pip check \
    && python -c "import ngsolve; assert ngsolve.__version__ == '6.2.2608'"
COPY requirements.precision.lock.txt /tmp/requirements.precision.lock.txt
RUN python -m pip install --no-cache-dir --timeout 120 --retries 5 --require-hashes --only-binary=:all: --no-deps -r /tmp/requirements.precision.lock.txt \
    && python -m pip check \
    && python -c "import ngsolve, mpmath, gmpy2; assert ngsolve.__version__ == '6.2.2608'; assert mpmath.__version__ == '1.3.0'; assert gmpy2.version() == '2.2.1'; assert mpmath.libmp.BACKEND == 'gmpy'"
WORKDIR /app
COPY src/ /app/src/
COPY runner/ /runner/
ENV PYTHONPATH=/app/src
