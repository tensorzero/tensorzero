-- Create the tensorzero schema for all TensorZero objects
CREATE SCHEMA IF NOT EXISTS tensorzero;

-- Move tensorzero_auth_api_key table to the tensorzero schema
ALTER TABLE tensorzero_auth_api_key SET SCHEMA tensorzero;
