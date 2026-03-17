CREATE TABLE tensorzero_auth_api_key (
    "id" BIGSERIAL PRIMARY KEY, -- for database only
    "organization" TEXT NOT NULL,
    "workspace" TEXT NOT NULL,
    "description" TEXT,
    "public_id" CHAR(12) NOT NULL, -- for users
    "hash" VARCHAR(255) NOT NULL,
    "disabled_at" TIMESTAMPTZ NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "uniq_public_id" UNIQUE ("public_id")
);
