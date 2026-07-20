# FORMA — LLM-powered multi-agent spectral analysis
#
# Build (default — GitHub reachable):
#   docker compose build
#
# Build (China / firewalled Docker — skip the git clone attempts):
#   REDROCK_SOURCE_DIR=/path/to/redrock \
#   REDROCK_TEMPLATES_DIR=/path/to/templates \
#   REDROCK_ARCHETYPES_DIR=/path/to/archetypes \
#     docker compose build --build-arg REDROCK_SKIP_GIT_CLONE=true
#
#   The three *_DIR vars are optional — templates inside the redrock repo
#   are auto-detected; archetypes are always optional.

FROM python:3.12-slim

# ── Layer 1: system packages ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    curl \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 2: Python dependencies (cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 3: Redrock redshift fitter ──
# git clone first (cached on success), local COPY --from is the fallback.
ARG REDROCK_SKIP_GIT_CLONE=false
ARG REDROCK_REPO=https://github.com/desihub/redrock
ARG REDROCK_TEMPLATES_REPO=https://github.com/desihub/redrock-templates
ARG REDROCK_ARCHETYPES_REPO=https://github.com/desihub/redrock-archetypes

RUN if [ "$REDROCK_SKIP_GIT_CLONE" = "true" ]; then \
      echo "==> REDROCK_SKIP_GIT_CLONE=true — skipping git clone."; \
      mkdir -p /opt/redrock && touch /opt/redrock/.git_clone_skipped; \
    else \
      echo "==> Trying git clone redrock..." \
      && git clone --depth 1 ${REDROCK_REPO} /opt/redrock \
      && cd /opt/redrock \
      && echo "==> Trying git clone templates..." \
      && git clone --depth 1 ${REDROCK_TEMPLATES_REPO} py/redrock/templates \
      && echo "    Done (GitHub)."; \
    fi

RUN if [ -f /opt/redrock/.git_clone_skipped ] || [ ! -d /opt/redrock/py/redrock ]; then \
      echo "==> git clone was skipped or failed — trying archetypes clone..." \
      && (git clone --depth 1 ${REDROCK_ARCHETYPES_REPO} /opt/redrock/redrock-archetypes || true); \
    else \
      echo "==> Trying git clone archetypes..." \
      && git clone --depth 1 ${REDROCK_ARCHETYPES_REPO} /opt/redrock/redrock-archetypes \
      && echo "    Done (GitHub)."; \
    fi

# ── Pull in local copies (fast, only used as fallback below) ──
COPY --from=local-redrock / /build/redrock/
COPY --from=local-templates / /build/templates/
COPY --from=local-archetypes / /build/archetypes/

# ── Fill gaps from local copies when git clone was skipped/failed ──
RUN if [ ! -d /opt/redrock/py/redrock ] || [ ! -f /opt/redrock/setup.py ] && [ ! -f /opt/redrock/pyproject.toml ]; then \
      echo "==> Redrock source missing — copying from REDROCK_SOURCE_DIR..." \
      && if [ -f /build/redrock/bin/rrdesi ] || [ -d /build/redrock/py/redrock ]; then \
           rm -rf /opt/redrock && cp -r /build/redrock /opt/redrock; \
         else \
           echo "ERROR: no redrock source (git clone skipped & no local copy)."; exit 1; \
         fi \
    fi

RUN if ! ls /opt/redrock/py/redrock/templates/rrtemplate-*.fits >/dev/null 2>&1; then \
      if ls /build/templates/rrtemplate-*.fits >/dev/null 2>&1; then \
        echo "==> Templates missing — copying from REDROCK_TEMPLATES_DIR..." \
        && mkdir -p /opt/redrock/py/redrock/templates \
        && cp -r /build/templates/. /opt/redrock/py/redrock/templates/; \
      else \
        echo "WARNING: templates not found. Set REDROCK_TEMPLATES_DIR to a directory containing rrtemplate-*.fits."; \
      fi \
    else \
      echo "==> Templates found."; \
    fi

RUN if [ ! -d /opt/redrock/redrock-archetypes ] || ! ls /opt/redrock/redrock-archetypes/rrarchetype-*.fits >/dev/null 2>&1; then \
      if ls /build/archetypes/rrarchetype-*.fits >/dev/null 2>&1; then \
        echo "==> Archetypes missing — copying from REDROCK_ARCHETYPES_DIR..." \
        && mkdir -p /opt/redrock/redrock-archetypes \
        && cp -r /build/archetypes/. /opt/redrock/redrock-archetypes/; \
      else \
        echo "==> Archetypes not found — continuing without them (optional)."; \
      fi \
    else \
      echo "==> Archetypes found."; \
    fi

# ── Install Redrock ──
RUN cd /opt/redrock \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir desiutil desispec

# ── Layer 4: FORMA source (most frequently changed) ──
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e ".[web]"

# ── Layer 5: entrypoint ──
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ── Environment defaults ──
ENV PYTHONUNBUFFERED=1
ENV REDROCK=true
ENV RR_TEMPLATE_DIR=/opt/redrock/py/redrock/templates
ENV ARCHETYPE_DIR=/opt/redrock/redrock-archetypes
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["cli"]
