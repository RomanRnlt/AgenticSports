-- Phase 7: Long-Term Intelligence
-- Macrocycle planning (4-52 week training blocks) and goal trajectory tracking.

-- ---------------------------------------------------------------------------
-- macrocycle_plans — multi-week training structure
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS macrocycle_plans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES profiles(id),
    name TEXT NOT NULL,
    periodization_model_name TEXT,
    total_weeks INT NOT NULL CHECK (total_weeks >= 1 AND total_weeks <= 52),
    start_date DATE NOT NULL,
    weeks JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'completed', 'archived')),
    evaluation_score INT CHECK (evaluation_score >= 0 AND evaluation_score <= 100),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, name)
);

CREATE INDEX idx_macrocycle_user ON macrocycle_plans(user_id);
CREATE INDEX idx_macrocycle_status ON macrocycle_plans(user_id, status);

ALTER TABLE macrocycle_plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users read own macrocycles" ON macrocycle_plans
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role full access macrocycles" ON macrocycle_plans
    FOR ALL USING (auth.role() = 'service_role');

-- ---------------------------------------------------------------------------
-- goal_trajectory_snapshots — append-only goal progress tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS goal_trajectory_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES profiles(id),
    goal_name TEXT NOT NULL,
    trajectory_status TEXT NOT NULL
        CHECK (trajectory_status IN ('on_track', 'ahead', 'behind', 'at_risk', 'insufficient_data')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    projected_outcome TEXT NOT NULL DEFAULT '',
    analysis TEXT NOT NULL DEFAULT '',
    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    risk_factors JSONB NOT NULL DEFAULT '[]'::jsonb,
    context_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_trajectory_user ON goal_trajectory_snapshots(user_id);
CREATE INDEX idx_trajectory_goal ON goal_trajectory_snapshots(user_id, goal_name);

ALTER TABLE goal_trajectory_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users read own trajectories" ON goal_trajectory_snapshots
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role full access trajectories" ON goal_trajectory_snapshots
    FOR ALL USING (auth.role() = 'service_role');
