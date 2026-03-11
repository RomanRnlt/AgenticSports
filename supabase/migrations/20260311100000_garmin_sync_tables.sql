-- Migration: Create health_daily_metrics table and fix activities upsert constraint
-- Required for Garmin sync to store daily health stats and deduplicate activities

-- ============================================================================
-- 1. health_daily_metrics — daily aggregated health metrics from wearables
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.health_daily_metrics (
  id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                 UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  date                    DATE NOT NULL,
  source                  TEXT NOT NULL DEFAULT 'garmin',

  -- Heart & HRV
  resting_heart_rate      INT,
  hrv_avg                 FLOAT,

  -- Sleep
  sleep_score             INT,
  sleep_duration_minutes  FLOAT,
  sleep_deep_minutes      FLOAT,
  sleep_light_minutes     FLOAT,
  sleep_rem_minutes       FLOAT,
  sleep_awake_minutes     FLOAT,

  -- Stress & Recovery
  stress_avg              INT,
  body_battery_high       INT,
  body_battery_low        INT,
  recovery_score          INT,

  -- Activity
  steps                   INT,
  active_calories         INT,
  total_calories          INT,
  floors_climbed          INT,
  intensity_minutes       INT,

  -- Respiratory
  spo2_avg                FLOAT,
  respiration_avg         FLOAT,

  -- Fitness
  vo2max                  FLOAT,

  -- Raw data
  raw_data                JSONB DEFAULT '{}'::jsonb,

  created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT health_daily_metrics_user_date_source_unique
    UNIQUE (user_id, date, source)
);

CREATE INDEX IF NOT EXISTS idx_health_daily_metrics_user_date
  ON public.health_daily_metrics(user_id, date DESC);

-- RLS
ALTER TABLE public.health_daily_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "hdm_select_own" ON public.health_daily_metrics
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "hdm_insert_own" ON public.health_daily_metrics
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);

CREATE POLICY "hdm_update_own" ON public.health_daily_metrics
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "hdm_service_all" ON public.health_daily_metrics
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- updated_at trigger
CREATE TRIGGER health_daily_metrics_updated_at
  BEFORE UPDATE ON public.health_daily_metrics
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Grant access
GRANT ALL ON public.health_daily_metrics TO authenticated, service_role;

-- ============================================================================
-- 2. Fix activities table — add missing columns and unique constraint for upsert
-- ============================================================================

-- Add missing columns used by garmin_sync
ALTER TABLE public.activities
  ADD COLUMN IF NOT EXISTS calories INT,
  ADD COLUMN IF NOT EXISTS training_effect FLOAT,
  ADD COLUMN IF NOT EXISTS vo2max_activity FLOAT;

-- Add a proper unique constraint for upsert (the partial index is not enough)
-- First drop the old partial index if it exists, then create the constraint
DROP INDEX IF EXISTS idx_activities_garmin_dedup;

ALTER TABLE public.activities
  ADD CONSTRAINT activities_user_garmin_id_unique
  UNIQUE (user_id, garmin_activity_id);
