# tensorzero/postgres — Postgres with pgvector and pg_cron
#
# Build args:
#   PGVECTOR_VERSION — pgvector version (default: 0.7.0)
#   PG_MAJOR         — major Postgres version (14, 15, 16, 17, 18)
#   DEBIAN_CODENAME  — Debian release (default: trixie)
#
# We derive from pgvector/pgvector which already bundles the vector extension
# on top of the official postgres image.

ARG PGVECTOR_VERSION=0.8.2
ARG PG_MAJOR=17
ARG DEBIAN_CODENAME=trixie
FROM pgvector/pgvector:${PGVECTOR_VERSION}-pg${PG_MAJOR}-${DEBIAN_CODENAME}

ARG PG_MAJOR

# Install pg_cron from Debian packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-${PG_MAJOR}-cron \
    && rm -rf /var/lib/apt/lists/*

# Configure postgresql.conf to preload pg_cron
RUN echo "shared_preload_libraries = 'pg_cron'" >> /usr/share/postgresql/postgresql.conf.sample

# The base postgres image creates a `postgres` user (UID 999)
USER postgres
