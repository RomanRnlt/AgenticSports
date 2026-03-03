-- ============================================================================
-- Remaining AgenticSports Schema (tables not created by initial partial run)
-- ============================================================================

-- Ensure pgvector is available
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Prerequisite: updated_at trigger function
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- METRIC DEFINITIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.metric_definitions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  formula     TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  unit        TEXT NOT NULL DEFAULT '',
  variables   JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

CREATE TRIGGER update_metric_definitions_updated_at
  BEFORE UPDATE ON public.metric_definitions
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE INDEX IF NOT EXISTS idx_metric_definitions_user
  ON public.metric_definitions(user_id);

ALTER TABLE public.metric_definitions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "metric_definitions_select_own" ON public.metric_definitions
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "metric_definitions_insert_own" ON public.metric_definitions
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "metric_definitions_update_own" ON public.metric_definitions
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "metric_definitions_service_all" ON public.metric_definitions
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- EVAL CRITERIA
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.eval_criteria (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  weight      FLOAT NOT NULL DEFAULT 1.0,
  formula     TEXT NOT NULL DEFAULT '',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

CREATE TRIGGER update_eval_criteria_updated_at
  BEFORE UPDATE ON public.eval_criteria
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE INDEX IF NOT EXISTS idx_eval_criteria_user
  ON public.eval_criteria(user_id);

ALTER TABLE public.eval_criteria ENABLE ROW LEVEL SECURITY;

CREATE POLICY "eval_criteria_select_own" ON public.eval_criteria
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "eval_criteria_insert_own" ON public.eval_criteria
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "eval_criteria_update_own" ON public.eval_criteria
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "eval_criteria_service_all" ON public.eval_criteria
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- SESSION SCHEMAS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.session_schemas (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  sport      TEXT NOT NULL,
  schema     JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, sport)
);

CREATE TRIGGER update_session_schemas_updated_at
  BEFORE UPDATE ON public.session_schemas
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE INDEX IF NOT EXISTS idx_session_schemas_user
  ON public.session_schemas(user_id);

ALTER TABLE public.session_schemas ENABLE ROW LEVEL SECURITY;

CREATE POLICY "session_schemas_select_own" ON public.session_schemas
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "session_schemas_insert_own" ON public.session_schemas
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "session_schemas_update_own" ON public.session_schemas
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "session_schemas_service_all" ON public.session_schemas
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- PERIODIZATION MODELS
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.periodization_models (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,
  phases     JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

CREATE TRIGGER update_periodization_models_updated_at
  BEFORE UPDATE ON public.periodization_models
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE INDEX IF NOT EXISTS idx_periodization_models_user
  ON public.periodization_models(user_id);

ALTER TABLE public.periodization_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "periodization_models_select_own" ON public.periodization_models
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "periodization_models_insert_own" ON public.periodization_models
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "periodization_models_update_own" ON public.periodization_models
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "periodization_models_service_all" ON public.periodization_models
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- PROACTIVE TRIGGER RULES
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.proactive_trigger_rules (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  condition       TEXT NOT NULL,
  action          TEXT NOT NULL,
  cooldown_hours  INT NOT NULL DEFAULT 24,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(user_id, name)
);

CREATE TRIGGER update_proactive_trigger_rules_updated_at
  BEFORE UPDATE ON public.proactive_trigger_rules
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE INDEX IF NOT EXISTS idx_proactive_trigger_rules_user
  ON public.proactive_trigger_rules(user_id);

ALTER TABLE public.proactive_trigger_rules ENABLE ROW LEVEL SECURITY;

CREATE POLICY "proactive_trigger_rules_select_own" ON public.proactive_trigger_rules
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "proactive_trigger_rules_insert_own" ON public.proactive_trigger_rules
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "proactive_trigger_rules_update_own" ON public.proactive_trigger_rules
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "proactive_trigger_rules_service_all" ON public.proactive_trigger_rules
  FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ============================================================================
-- MATCH BELIEFS RPC (idempotent)
-- ============================================================================

CREATE OR REPLACE FUNCTION public.match_beliefs(
  p_user_id       UUID,
  p_embedding      extensions.vector(768),
  p_match_count    INT DEFAULT 5,
  p_min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
  id         UUID,
  text       TEXT,
  category   TEXT,
  confidence FLOAT,
  similarity FLOAT
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    b.id,
    b.text,
    b.category,
    b.confidence,
    1 - (b.embedding <=> p_embedding) AS similarity
  FROM public.beliefs b
  WHERE b.user_id = p_user_id
    AND b.active = true
    AND b.embedding IS NOT NULL
    AND b.confidence >= p_min_confidence
  ORDER BY b.embedding <=> p_embedding
  LIMIT p_match_count;
END;
$$;


-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT ALL ON public.metric_definitions    TO authenticated, service_role;
GRANT ALL ON public.eval_criteria         TO authenticated, service_role;
GRANT ALL ON public.session_schemas       TO authenticated, service_role;
GRANT ALL ON public.periodization_models  TO authenticated, service_role;
GRANT ALL ON public.proactive_trigger_rules TO authenticated, service_role;
