CREATE TABLE tensorzero_auth_api_key (
"id" BIGSERIAL PRIMARY KEY,
"organization" TEXT NOT NULL,
"workspace" TEXT NOT NULL,
"description" TEXT,
"short_id" CHAR(12) NOT NULL,
"hash" VARCHAR(255) NOT NULL,
"disabled_at" TIMESTAMPTZ NULL,
"created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
"updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
CONSTRAINT "uniq_short_id" UNIQUE ("short_id")
);