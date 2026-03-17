# tensorzero/postgres — Postgres with pgvector and pg_cron
#
# Build args:
#   PG_MAJOR         — major Postgres version (14, 15, 16, 17, 18)
#   PG_MINOR         — full patch version (e.g. 17.9)
#   PGVECTOR_VERSION — pgvector version (default: 0.8.2)
#   DEBIAN_CODENAME  — Debian release (default: trixie)
#
# We base on the official postgres image (pinned to the patch version) and
# build pgvector from source so the published tag accurately reflects the
# Postgres patch version.

ARG PG_MAJOR=17
ARG PG_MINOR=17.9
ARG DEBIAN_CODENAME=trixie
FROM postgres:${PG_MINOR}-${DEBIAN_CODENAME}

ARG PG_MAJOR
ARG PGVECTOR_VERSION=0.8.2

# Install pgvector from source and pg_cron from Debian packages
# pgvector installation script is copied from
# https://github.com/pgvector/pgvector/blob/master/Dockerfile
RUN apt-get update && \
    apt-mark hold locales && \
    apt-get install -y --no-install-recommends build-essential postgresql-server-dev-${PG_MAJOR} ca-certificates git postgresql-${PG_MAJOR}-cron && \
    git clone --branch v${PGVECTOR_VERSION} --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector && \
    cd /tmp/pgvector && \
    make clean && \
    make OPTFLAGS="" && \
    make install && \
    mkdir -p /usr/share/doc/pgvector && \
    cp LICENSE README.md /usr/share/doc/pgvector && \
    rm -r /tmp/pgvector && \
    apt-get remove -y build-essential postgresql-server-dev-${PG_MAJOR} ca-certificates git && \
    apt-get autoremove -y && \
    apt-mark unhold locales && \
    rm -rf /var/lib/apt/lists/*

# Configure postgresql.conf to preload pg_cron
RUN echo "shared_preload_libraries = 'pg_cron'" >> /usr/share/postgresql/postgresql.conf.sample

# The base postgres image creates a `postgres` user (UID 999)
USER postgres
