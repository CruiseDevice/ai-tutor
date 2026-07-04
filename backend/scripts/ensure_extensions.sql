-- One-time DBA / infra script to ensure required Postgres extensions exist.
-- Run this against the target database before starting the application:
--
--   psql -U postgres -d studyfetch -f scripts/ensure_extensions.sql
--
-- In managed Postgres (RDS, Supabase, etc.) the application role may lack
-- CREATE privilege, so this should be executed by an administrative role.

CREATE EXTENSION IF NOT EXISTS vector;
