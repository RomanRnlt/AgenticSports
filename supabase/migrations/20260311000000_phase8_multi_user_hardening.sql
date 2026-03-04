-- Phase 8a: Multi-User RLS Hardening
-- Ensures consistent service_role bypass policies on all tables.
-- Most tables already have service_role policies from agenticsports_schema.sql;
-- this migration adds the missing one (daily_usage) and ensures GRANTS are complete.

-- ============================================================================
-- 1. Service-role policy on daily_usage (the only table missing one)
-- ============================================================================
-- The init_schema.sql created daily_usage with only a user-own policy.
-- The agenticsports_schema.sql did not add a service_role policy for it.

CREATE POLICY "daily_usage_service_all" ON public.daily_usage
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ============================================================================
-- 2. GRANTS — ensure daily_usage has explicit grants (consistency)
-- ============================================================================

GRANT ALL ON public.daily_usage TO authenticated, service_role;

-- ============================================================================
-- 3. user_id Index Audit — add missing indexes for RLS filter performance
-- ============================================================================
-- daily_usage: PK is (user_id, usage_date) so user_id is already indexed.
-- session_messages: has idx on (session_id, id) and (user_id, created_at).
-- All other tables already have user_id indexes.
-- No new indexes needed — all tables with user_id have proper indexes.

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Summary:
--   - Added service_role policy on daily_usage (the only missing one)
--   - Added GRANT ALL on daily_usage for authenticated + service_role
--   - Verified all user_id columns have indexes (none missing)
