CREATE TABLE IF NOT EXISTS provider_tokens (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    token_data JSONB NOT NULL,
    provider_user_id TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'expired', 'revoked')),
    last_sync_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, provider)
);

CREATE INDEX idx_provider_tokens_user ON provider_tokens(user_id);
CREATE INDEX idx_provider_tokens_provider ON provider_tokens(user_id, provider);

ALTER TABLE provider_tokens ENABLE ROW LEVEL SECURITY;

CREATE POLICY "provider_tokens_select_own" ON provider_tokens
  FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "provider_tokens_service_all" ON provider_tokens
  FOR ALL TO service_role USING (true) WITH CHECK (true);

GRANT ALL ON provider_tokens TO authenticated, service_role;
