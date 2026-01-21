-- Move all tensorzero-core objects to the tensorzero schema

-- Create the schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS tensorzero;

-- Move functions FIRST (before the type they reference)
-- Functions reference `calculated_bucket_state` without schema qualification.
-- They must be moved while the type is still in public schema so the reference resolves.
ALTER FUNCTION _calculate_refilled_state(BIGINT, TIMESTAMPTZ, BIGINT, BIGINT, INTERVAL) SET SCHEMA tensorzero;
ALTER FUNCTION get_resource_bucket_balance(TEXT, BIGINT, BIGINT, INTERVAL) SET SCHEMA tensorzero;
ALTER FUNCTION consume_multiple_resource_tickets(TEXT[], BIGINT[], BIGINT[], BIGINT[], INTERVAL[]) SET SCHEMA tensorzero;
ALTER FUNCTION return_multiple_resource_tickets(TEXT[], BIGINT[], BIGINT[], BIGINT[], INTERVAL[]) SET SCHEMA tensorzero;

-- Move the custom type (after functions, since they have a dependency on it)
ALTER TYPE calculated_bucket_state SET SCHEMA tensorzero;

-- Move tables
ALTER TABLE resource_bucket SET SCHEMA tensorzero;
ALTER TABLE variant_by_episode SET SCHEMA tensorzero;
