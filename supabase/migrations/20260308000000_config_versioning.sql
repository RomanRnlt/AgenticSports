-- Phase 4c: Config Store Versioning & Garbage Collection
-- Adds temporal versioning to all agent_config tables so the agent
-- can track config evolution and garbage-collect stale entries.

-- metric_definitions
ALTER TABLE metric_definitions
  ADD COLUMN IF NOT EXISTS valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS valid_until  TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_by UUID;

-- eval_criteria
ALTER TABLE eval_criteria
  ADD COLUMN IF NOT EXISTS valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS valid_until  TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_by UUID;

-- session_schemas
ALTER TABLE session_schemas
  ADD COLUMN IF NOT EXISTS valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS valid_until  TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_by UUID;

-- periodization_models
ALTER TABLE periodization_models
  ADD COLUMN IF NOT EXISTS valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS valid_until  TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_by UUID;

-- proactive_trigger_rules
ALTER TABLE proactive_trigger_rules
  ADD COLUMN IF NOT EXISTS valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS valid_until  TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_by UUID;

-- Index for GC queries: quickly find active vs archived configs
CREATE INDEX IF NOT EXISTS idx_metric_definitions_active
  ON metric_definitions (user_id) WHERE valid_until IS NULL;

CREATE INDEX IF NOT EXISTS idx_eval_criteria_active
  ON eval_criteria (user_id) WHERE valid_until IS NULL;

CREATE INDEX IF NOT EXISTS idx_proactive_trigger_rules_active
  ON proactive_trigger_rules (user_id) WHERE valid_until IS NULL;
