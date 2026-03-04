-- Phase 6: Product Recommendations
-- Agent-driven product recommendations with affiliate link support.

CREATE TABLE IF NOT EXISTS product_recommendations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES profiles(id),
    session_id UUID,
    plan_id UUID,
    product_name TEXT NOT NULL,
    product_description TEXT,
    image_url TEXT,
    price NUMERIC(10,2),
    currency TEXT DEFAULT 'EUR',
    product_url TEXT,
    affiliate_url TEXT,
    affiliate_provider TEXT,
    reason TEXT NOT NULL,
    category TEXT,
    sport TEXT,
    search_query TEXT,
    source TEXT DEFAULT 'llm',
    clicked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_prod_rec_user ON product_recommendations(user_id);
CREATE INDEX idx_prod_rec_session ON product_recommendations(session_id);
CREATE INDEX idx_prod_rec_plan ON product_recommendations(plan_id);

ALTER TABLE product_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users read own recommendations" ON product_recommendations
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role full access" ON product_recommendations
    FOR ALL USING (auth.role() = 'service_role');
