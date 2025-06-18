ARG CLICKHOUSE_VERSION=24.12
FROM clickhouse/clickhouse-server:${CLICKHOUSE_VERSION} AS load-fixtures
COPY --from=fixtures_gateway /usr/local/bin/gateway /usr/local/bin/gateway
WORKDIR /ui/fixtures
COPY . .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DATABASE_NAME=tensorzero_ui_fixtures
ENV CLICKHOUSE_HOST=localhost
ENV CLICKHOUSE_PORT=9000
ENV CLICKHOUSE_USER=chuser
ENV CLICKHOUSE_PASSWORD=chpassword
ENV CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
ENV TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures
RUN ./start-server-and-load-fixtures.sh

FROM clickhouse/clickhouse-server:${CLICKHOUSE_VERSION}
COPY --from=load-fixtures /var/lib/clickhouse/ /var/lib/clickhouse/
ENTRYPOINT ["/entrypoint.sh"]