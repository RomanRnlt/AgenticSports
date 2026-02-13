"""Conversation engine: the core pipeline for chat-based coaching.

Implements the 5-level context management system with dynamic token budgets,
belief extraction via single LLM call, Mem0 two-phase compare/merge,
session storage (JSONL), rolling summaries, and session consolidation.
"""

import json
from datetime import datetime
from pathlib import Path

from google import genai

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client
from src.memory.user_model import UserModel, EMBEDDING_MODEL

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

# ── Dynamic Token Budgets (ACL 2025: Token-Budget-Aware Reasoning) ────
# Approximate char limits (1 token ~ 4 chars)
TOKEN_BUDGETS = {
    "onboarding": {"system": 6000, "model": 1200, "activity": 0,    "cross": 0,    "rolling": 1200, "recent": 8000},
    "early":      {"system": 4800, "model": 4000, "activity": 3000, "cross": 2000, "rolling": 2000, "recent": 10000},
    "ongoing":    {"system": 3200, "model": 8000, "activity": 4000, "cross": 4000, "rolling": 3200, "recent": 16000},
    "planning":   {"system": 3200, "model": 10000,"activity": 4000, "cross": 6000, "rolling": 2000, "recent": 12000},
}

# Rolling summary is regenerated every N new turns
ROLLING_SUMMARY_INTERVAL = 5
# Number of recent turns to keep verbatim
RECENT_TURNS_COUNT = 10

# JSON response schema for Gemini structured output — forces consistent field names.
CONVERSATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "response_text": {"type": "STRING"},
        "extracted_beliefs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "category": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                    "reasoning": {"type": "STRING"},
                },
                "required": ["text", "category", "confidence"],
            },
        },
        "structured_core_updates": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING", "nullable": True},
                "sports": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "nullable": True,
                },
                "goal": {
                    "type": "OBJECT",
                    "properties": {
                        "event": {"type": "STRING", "nullable": True},
                        "target_date": {"type": "STRING", "nullable": True},
                        "target_time": {"type": "STRING", "nullable": True},
                    },
                    "nullable": True,
                },
                "fitness": {
                    "type": "OBJECT",
                    "properties": {
                        "estimated_vo2max": {"type": "NUMBER", "nullable": True},
                        "threshold_pace_min_km": {"type": "STRING", "nullable": True},
                        "weekly_volume_km": {"type": "NUMBER", "nullable": True},
                    },
                    "nullable": True,
                },
                "constraints": {
                    "type": "OBJECT",
                    "properties": {
                        "training_days_per_week": {"type": "INTEGER", "nullable": True},
                        "max_session_minutes": {
                            "type": "INTEGER",
                            "nullable": True,
                            "description": "MAXIMUM session duration in minutes across the entire week INCLUDING weekends. If weekday=90min, weekend=3h, set to 180.",
                        },
                    },
                    "nullable": True,
                },
            },
        },
        "trigger_cycle": {"type": "BOOLEAN"},
        "cycle_reason": {"type": "STRING", "nullable": True},
        "onboarding_complete": {"type": "BOOLEAN"},
    },
    "required": [
        "response_text", "extracted_beliefs", "structured_core_updates",
        "trigger_cycle", "onboarding_complete",
    ],
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_llm_response(raw: dict) -> dict:
    """Normalize LLM JSON response to canonical key names.

    Gemini returns varying key names and structures depending on the prompt.
    This maps all known variants to the expected schema:
      response_text, extracted_beliefs, structured_core_updates,
      trigger_cycle, cycle_reason, onboarding_complete
    """
    result: dict = {}

    # --- response_text ---
    result["response_text"] = (
        raw.get("response_text")
        or raw.get("response")
        or raw.get("structured_coach_response")
        or raw.get("natural_language_response")
        or raw.get("coach_response")
        or raw.get("coaching_response")
        or raw.get("text")
        or "I'm here to help with your training."
    )

    # --- extracted_beliefs ---
    beliefs = raw.get("extracted_beliefs")
    if beliefs is None:
        beliefs = raw.get("athlete_beliefs") or raw.get("beliefs") or []
        if not beliefs:
            scu = raw.get("structured_core_updates", {})
            if isinstance(scu, dict):
                beliefs = scu.pop("athlete_beliefs", None) or scu.pop("beliefs", None) or []

    normalized_beliefs = _normalize_beliefs(beliefs if isinstance(beliefs, list) else [])

    # --- structured_core_updates ---
    scu = raw.get("structured_core_updates", {})
    if isinstance(scu, dict):
        scu.pop("athlete_beliefs", None)
        scu.pop("beliefs", None)
        scu.pop("training_cycle_trigger", None)
    else:
        scu = {}

    # Extract synthetic beliefs from activity_preferences / training_preferences
    for prefs_key in ("activity_preferences", "training_preferences"):
        prefs = scu.pop(prefs_key, None)
        if isinstance(prefs, dict):
            synth_beliefs, extra_core = _extract_activity_preference_data(prefs)
            normalized_beliefs.extend(synth_beliefs)
            scu.update(extra_core)

    result["extracted_beliefs"] = normalized_beliefs
    result["structured_core_updates"] = _flatten_core_updates(scu)

    # --- trigger_cycle ---
    tc = raw.get("trigger_cycle")
    if tc is None:
        tct = raw.get("training_cycle_trigger", {})
        if isinstance(tct, dict):
            tc = tct.get("trigger", False)
        else:
            tc = bool(tct)
    result["trigger_cycle"] = bool(tc)

    # --- cycle_reason ---
    reason = raw.get("cycle_reason")
    if reason is None:
        tct = raw.get("training_cycle_trigger", {})
        if isinstance(tct, dict):
            reason = tct.get("reason")
    result["cycle_reason"] = reason

    # --- onboarding_complete ---
    oc = raw.get("onboarding_complete")
    if oc is None:
        oc = raw.get("onboarding_status", {})
        if isinstance(oc, dict):
            oc = oc.get("complete", False)
    result["onboarding_complete"] = bool(oc) if oc is not None else False

    return result


def _normalize_beliefs(beliefs: list) -> list[dict]:
    """Normalize a list of belief dicts to canonical format."""
    normalized = []
    for b in beliefs:
        text = b.get("text") or b.get("belief") or b.get("value") or b.get("statement", "")
        if isinstance(text, (list, tuple)):
            text = ", ".join(str(x) for x in text)
        elif not isinstance(text, str):
            text = str(text) if text else ""
        if text:
            normalized.append({
                "text": text,
                "category": b.get("category", "preference"),
                "confidence": b.get("confidence", 0.7),
                "reasoning": b.get("reasoning") or b.get("reason", ""),
            })
    return normalized


def _extract_activity_preference_data(prefs: dict) -> tuple[list[dict], dict]:
    """Extract beliefs and core updates from Gemini's activity_preferences section.

    Gemini often returns an activity_preferences dict instead of proper beliefs:
      {"preferred_sports": [...], "preferred_run_sessions_per_week": 3,
       "has_zwift_access": true, ...}

    Returns (synthetic_beliefs, extra_core_updates).
    """
    beliefs: list[dict] = []
    core: dict = {}

    # preferred_sports → top-level sports
    sports = prefs.get("preferred_sports")
    if isinstance(sports, list) and sports:
        core["sports"] = sports

    # Sum up session counts → constraints.training_days_per_week
    session_keys = [k for k in prefs if "sessions_per_week" in k and k != "max_sessions_per_week"]
    total_sessions = 0
    for k in session_keys:
        v = prefs.get(k)
        if isinstance(v, (int, float)):
            total_sessions += int(v)
            sport = k.replace("preferred_", "").replace("_sessions_per_week", "").replace("_", " ")
            beliefs.append({
                "text": f"Wants {int(v)}x {sport} per week",
                "category": "scheduling",
                "confidence": 0.85,
                "reasoning": "Extracted from stated training plan",
            })

    # Use max_sessions_per_week if provided, else sum of individual counts
    max_sessions = prefs.get("max_sessions_per_week")
    if isinstance(max_sessions, (int, float)):
        core["constraints"] = {"training_days_per_week": min(int(max_sessions), 7)}
    elif total_sessions > 0:
        core["constraints"] = {"training_days_per_week": min(total_sessions, 7)}

    # Generic boolean preferences → beliefs (e.g., has_zwift_access, has_indoor_trainer)
    for key, value in prefs.items():
        if isinstance(value, bool) and value and key.startswith("has_"):
            label = key.replace("has_", "").replace("_", " ")
            beliefs.append({
                "text": f"Has {label}",
                "category": "preference",
                "confidence": 0.9,
                "reasoning": f"Extracted from {key}",
            })

    return beliefs, core


def _normalize_pace(pace_str: str) -> str:
    """Normalize pace format to MM:SS.

    Gemini may return "0:04:16" or "00:04:16" (H:MM:SS) instead of "04:16" (MM:SS).
    """
    if not isinstance(pace_str, str):
        return str(pace_str)
    parts = pace_str.split(":")
    if len(parts) == 3 and parts[0] in ("0", "00"):
        return f"{parts[1]}:{parts[2]}"
    return pace_str




# Mapping of Gemini's non-standard field names to canonical dot-notation paths.
_CORE_FIELD_ALIASES: dict[str, str] = {
    # goal fields — generic nested→dot mappings
    "goal.event": "goal.event",
    "goals.event": "goal.event",
    "event": "goal.event",
    "goal_event": "goal.event",
    "goals.primary_race_target": "goal.event",
    "primary_race_target": "goal.event",
    "motivation.race_goal": "goal.event",
    "motivation.event": "goal.event",
    "goal.target_date": "goal.target_date",
    "goals.target_date": "goal.target_date",
    "target_date": "goal.target_date",
    "motivation.target_race_date": "goal.target_date",
    "motivation.target_date": "goal.target_date",
    "goal.target_time": "goal.target_time",
    "goals.target_time": "goal.target_time",
    "goals.target_performance": "goal.target_time",
    "target_time": "goal.target_time",
    "target_performance": "goal.target_time",
    "motivation.race_target_time": "goal.target_time",
    "motivation.target_time": "goal.target_time",
    "goal.goal_type": "goal.goal_type",
    "goals.goal_type": "goal.goal_type",
    "goal_type": "goal.goal_type",
    # fitness fields
    "fitness.estimated_vo2max": "fitness.estimated_vo2max",
    "estimated_vo2max": "fitness.estimated_vo2max",
    "fitness.threshold_pace_min_km": "fitness.threshold_pace_min_km",
    "threshold_pace_min_km": "fitness.threshold_pace_min_km",
    "fitness.weekly_volume_km": "fitness.weekly_volume_km",
    "weekly_volume_km": "fitness.weekly_volume_km",
    "fitness.resting_hr": "fitness.resting_hr",
    "resting_hr": "fitness.resting_hr",
    "fitness.max_hr": "fitness.max_hr",
    "max_hr": "fitness.max_hr",
    "fitness.ftp": "fitness.ftp",
    "ftp": "fitness.ftp",
    # constraint fields
    "constraints.training_days_per_week": "constraints.training_days_per_week",
    "training_days_per_week": "constraints.training_days_per_week",
    "constraints.max_session_minutes": "constraints.max_session_minutes",
    "max_session_minutes": "constraints.max_session_minutes",
    "constraints.available_sports": "constraints.available_sports",
    "available_sports": "constraints.available_sports",
    "constraints.max_weekday_minutes": "constraints.max_weekday_minutes",
    "max_weekday_minutes": "constraints.max_weekday_minutes",
    "constraints.max_weekend_minutes": "constraints.max_weekend_minutes",
    "max_weekend_minutes": "constraints.max_weekend_minutes",
    # top-level fields
    "sports": "sports",
    "preferred_sports": "sports",
    "name": "name",
    "athlete_name": "name",
}


def _flatten_core_updates(scu: dict) -> dict:
    """Flatten nested structured_core_updates into dot-notation paths.

    Recursively flattens nested dicts and maps non-standard field names
    to canonical paths via _CORE_FIELD_ALIASES.
    """
    flat: dict = {}

    for key, value in scu.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                dotted = f"{key}.{sub_key}"
                canonical = _CORE_FIELD_ALIASES.get(dotted) or _CORE_FIELD_ALIASES.get(sub_key)
                if canonical:
                    if "pace" in canonical and isinstance(sub_value, str):
                        sub_value = _normalize_pace(sub_value)
                    flat[canonical] = sub_value
                else:
                    flat[dotted] = sub_value
        else:
            canonical = _CORE_FIELD_ALIASES.get(key)
            if canonical:
                if "pace" in canonical and isinstance(value, str):
                    value = _normalize_pace(value)
                flat[canonical] = value
            else:
                flat[key] = value

    return flat


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, cutting at last newline before limit."""
    if not text or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars // 2:
        return cut[:last_nl] + "\n[...truncated]"
    return cut + "\n[...truncated]"


# ── System Prompts ──────────────────────────────────────────────

CONVERSATION_SYSTEM_PROMPT = """\
You are ReAgt, an experienced endurance sports coach. You help athletes with running, cycling, swimming, and general fitness through natural conversation.

TODAY'S DATE: {today}

## Your Core Behaviors
- Be warm, knowledgeable, and data-driven
- Ask clarifying questions when you need more information
- Reference specific data points when available (pace, HR, distance, etc.)
- Be concise but thorough -- match the athlete's communication style
- NEVER make up training data. Only reference data you have been given.
- When the athlete asks a question, ANSWER it with your coaching knowledge. Do not deflect by asking for more data.

## Training Data Usage
You have access to a TRAINING DATA section in the conversation context. It contains:
- LAST SESSION: Detailed metrics from the most recent training session, plus brief summaries of the latest session per sport
- THIS WEEK: 7-day aggregated summary with per-sport breakdown (sessions, distance, pace/speed, HR)
- 4-WEEK TRENDS: Weekly aggregates showing volume and training load trends over the past month
- PLAN vs ACTUAL: If a training plan exists, weekly compliance comparison

CRITICAL RULES for training data:
1. All numbers in TRAINING DATA are pre-computed and accurate. NEVER recalculate, estimate, or round them differently. Present them exactly as shown.
2. When the athlete asks about a specific workout ("How was my last run?"), reference the exact data -- pace, HR, distance, zones, TRIMP.
3. When the athlete asks about their week, use the THIS WEEK section for per-sport breakdown and totals.
4. When the athlete asks pattern questions ("Why am I tired?", "Am I improving?"), cross-reference 4-WEEK TRENDS with the weekly data. Look for volume increases, TRIMP spikes, and HR changes.
5. If TRAINING DATA shows "No training data available", do not fabricate training data. Tell the athlete you don't have activity data yet.
6. Zone distribution percentages tell you intensity: high Z1-Z2 = easy/recovery, high Z3 = tempo/threshold, high Z4-Z5 = high intensity.
7. TRIMP (Training Impulse) is a load metric: higher = more training stress. Compare week-over-week to assess load progression.
8. When plan-vs-actual data is present, note the compliance rate but consider cross-training as valuable even if not in the plan.

## What You Must Do With EVERY User Message
Analyze the message and decide:
1. What natural coaching response to give
2. Whether any new beliefs about the athlete can be extracted
3. Whether fitness metrics can be DERIVED from any performance data mentioned (race times, FTP, known paces) → set structured_core_updates for fitness.estimated_vo2max and fitness.threshold_pace_min_km
4. Whether a training cycle (plan generation/assessment) should be triggered
5. Whether onboarding information gathering is complete
6. Whether the TRAINING DATA context contains relevant information to reference in your response

## Belief Extraction Rules
Extract beliefs when the user reveals:
- Training preferences (scheduling, intensity, types they enjoy)
- Physical constraints (injuries, health conditions, limitations)
- Goals and motivations (what drives them, race targets)
- Fitness data (recent performances, current ability level)
- Personal context (work schedule, family, stress, sleep)
- History (past races, training background, previous injuries)

Do NOT extract trivial greetings or conversation fillers as beliefs.

IMPORTANT - Belief categories must be ACCURATE. Use the correct category for each belief:
- "scheduling" for availability, training days, time constraints, weekday/weekend differences
- "constraint" for physical limitations, time limits, equipment constraints
- "fitness" for race times, performance data, VO2max, HR data, FTP, paces
- "motivation" for goals, race targets, what drives them
- "history" for past performances, training background, injuries
- "preference" for sport preferences, workout type preferences, coaching style preferences
- "physical" for body metrics, injuries, health conditions
- "personality" for communication style, coaching relationship preferences

Do NOT default everything to "preference". Choose the MOST SPECIFIC category.

## Fitness Derivation (Critical)
When the athlete mentions ANY performance data, derive fitness metrics:
- Race times (e.g., "10km in 42:30", "half marathon in 1:42") → estimate VO2max and threshold pace
- Cycling FTP (e.g., "FTP 260W at 75kg") → estimate VO2max from FTP (approximate: VO2max ≈ FTP_per_kg * 10.8 + 7)
- Known paces or power data → derive threshold estimates
- Put these estimates into structured_core_updates as fitness.estimated_vo2max and fitness.threshold_pace_min_km
- For cyclists without running data, threshold_pace_min_km can remain null, but VO2max should be estimated from cycling power
- Also estimate appropriate HR zones if resting/max HR is known

## Date Handling (Critical)
When the athlete mentions dates, ALWAYS convert to ISO format (YYYY-MM-DD) for structured_core_updates:
- "August" or "im August" → "2026-08-15" (mid-month if no specific date)
- "September 2026" → "2026-09-15"
- "Q3 2026" → "2026-09-01"
- Always use the CURRENT YEAR ({year}) or later. Never use past years.

## Constraints Handling
Capture the FULL picture of constraints:
- Set max_session_minutes to the MAXIMUM session duration across the whole week (including weekends)
- If the athlete has different limits for weekdays vs weekends, the weekday limit will be captured in beliefs
- Example: weekday 90min, weekend 3h → max_session_minutes=180, belief captures "weekday limit is 90 minutes"

## Triggering a Training Cycle
Trigger a cycle when:
- The athlete asks for a new plan or plan adjustment
- A significant constraint has changed (injury, schedule change)
- Enough onboarding info is gathered for initial plan generation
- The athlete reports something that warrants re-assessment

Do NOT trigger for casual conversation, status updates, or simple questions.

## Response Format
You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

{{
    "response_text": "Your natural coaching response to the athlete",
    "extracted_beliefs": [
        {{
            "text": "One-sentence belief statement about the athlete",
            "category": "scheduling|preference|constraint|history|motivation|physical|fitness|personality",
            "confidence": 0.5-0.95,
            "reasoning": "Brief explanation of why you extracted this"
        }}
    ],
    "structured_core_updates": {{
        "field.path": "value"
    }},
    "trigger_cycle": false,
    "cycle_reason": null,
    "onboarding_complete": false
}}

Rules for the JSON:
- extracted_beliefs: array of 0+ beliefs. Empty array if nothing to extract.
- structured_core_updates: object mapping dot-notation field paths to values. Empty object if no updates.
  Valid fields: name, sports, goal.event, goal.target_date, goal.target_time,
  fitness.estimated_vo2max, fitness.threshold_pace_min_km, fitness.weekly_volume_km,
  constraints.training_days_per_week, constraints.max_session_minutes, constraints.available_sports
- trigger_cycle: true only when a state machine cycle should run
- cycle_reason: short explanation if trigger_cycle is true, null otherwise
- onboarding_complete: true ONLY when you have gathered enough information to generate a training plan
  (minimum: sports, goal event, target date, training days per week)
"""

ROLLING_SUMMARY_PROMPT = """\
Summarize this coaching conversation so far. Focus on:
- Training decisions made
- Athlete's reported state (fatigue, motivation, injuries)
- Specific data points mentioned (pace, distance, HR, times)
- Open questions or unresolved topics
- Key preferences or constraints revealed

Be concise. Maximum 300 words. Preserve specific numbers and dates.
"""

SESSION_SUMMARY_PROMPT = """\
Generate a structured summary of this coaching session.

You MUST respond with ONLY a valid JSON object:
{
    "summary": "2-4 sentence overview of what was discussed and decided",
    "key_topics": ["topic1", "topic2"],
    "decisions_made": ["decision1", "decision2"],
    "athlete_mood": "brief mood assessment",
    "next_session_context": "What to follow up on next time"
}
"""

BELIEF_MERGE_PROMPT = """\
You are comparing a NEW candidate belief against EXISTING beliefs about an athlete.

Decide what operation to perform:
- ADD: The candidate is genuinely new information not captured by any existing belief
- UPDATE: The candidate refines, strengthens, or extends an existing belief (specify which one)
- DELETE: The candidate contradicts an existing belief, making it outdated (specify which one to supersede)
- NOOP: The candidate is already captured by an existing belief and adds nothing new

You MUST respond with ONLY a valid JSON object:
{
    "operation": "ADD|UPDATE|DELETE|NOOP",
    "target_belief_id": "id of existing belief to update/delete, or null for ADD/NOOP",
    "new_text": "updated belief text for UPDATE, or null",
    "new_confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision"
}
"""


class ConversationEngine:
    """Core conversation pipeline with 5-level context management."""

    def __init__(
        self,
        user_model: UserModel | None = None,
        sessions_dir: Path | None = None,
        data_dir: Path | None = None,
    ):
        self.user_model = user_model or UserModel.load_or_create()
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._sessions_dir = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

        # Current session state
        self._session_file: Path | None = None
        self._session_id: str | None = None
        self._turn_count: int = 0
        self._turns_since_summary: int = 0
        self._rolling_summary: str = ""
        self._cycle_triggered: bool = False

        # BM25 search index (lazy-built)
        self._bm25_index = None
        self._bm25_corpus: list[dict] | None = None

    # ── Session Lifecycle ────────────────────────────────────────

    def start_session(self) -> str:
        """Start a new conversation session. Returns the session ID."""
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._session_id = f"session_{ts}"
        self._session_file = self._sessions_dir / f"{self._session_id}.jsonl"
        self._turn_count = 0
        self._turns_since_summary = 0
        self._rolling_summary = ""
        self._cycle_triggered = False

        # Reset BM25 index (will be rebuilt on demand)
        self._bm25_index = None
        self._bm25_corpus = None

        # Update last interaction
        self.user_model.meta["last_interaction"] = _now_iso()

        return self._session_id

    def end_session(self) -> dict | None:
        """End session with full consolidation lifecycle.

        Returns the session summary dict, or None if no turns occurred.
        """
        if not self._session_file or self._turn_count == 0:
            return None

        summary = None

        # 1. Generate session summary via LLM
        try:
            summary = self._generate_session_summary()
        except Exception:
            summary = {
                "session_id": self._session_id,
                "date": datetime.now().date().isoformat(),
                "duration_turns": self._turn_count,
                "summary": "Session ended without summary generation.",
                "key_topics": [],
                "decisions_made": [],
                "athlete_mood": "unknown",
                "next_session_context": "",
            }

        # Save summary
        if summary:
            summary_path = self._sessions_dir / f"summary_{self._session_id.replace('session_', '')}.json"
            summary_path.write_text(json.dumps(summary, indent=2))

        # 2. Consolidate: promote stable session beliefs to global
        for belief in self.user_model.beliefs:
            if belief["active"] and belief["durability"] == "session":
                if belief["confidence"] >= 0.7:
                    belief["durability"] = "global"

        # 3. Forget: prune stale beliefs
        self.user_model.prune_stale_beliefs()

        # 4. Update session count and save
        self.user_model.meta["sessions_completed"] = (
            self.user_model.meta.get("sessions_completed", 0) + 1
        )
        self.user_model.save()

        # Reset session state
        self._session_file = None
        self._session_id = None

        return summary

    def inject_startup_greeting(self, greeting: str) -> None:
        """Inject the startup greeting as the first agent turn in session history.

        This ensures the LLM has context about what the coach said on startup,
        providing conversation continuity.

        Args:
            greeting: The greeting text displayed to the user on startup.
        """
        if not greeting:
            return

        self._append_to_session(
            role="agent",
            content=greeting,
            metadata={"artifacts": ["startup_greeting"]},
        )
        self._turn_count += 1

    # ── Main Message Processing ──────────────────────────────────

    def process_message(self, user_message: str) -> str:
        """Process a user message through the full conversation pipeline.

        1. Append user message to session JSONL
        2. Build 5-level context prompt
        3. Call LLM for structured response
        4. Process extracted beliefs through compare/merge
        5. Apply structured_core updates
        6. Append agent response to session JSONL
        7. Return response text
        """
        if not self._session_file:
            self.start_session()

        # 1. Log user message
        self._append_to_session("user", user_message)
        self._turn_count += 1
        self._turns_since_summary += 1

        # 1.5. Classify message for routing (v2.0 audit finding #8)
        route = "general_chat"
        try:
            from src.agent.router import classify_message, get_budget_overrides
            route = classify_message(user_message)
        except Exception:
            pass  # Fall back to general_chat on any routing error

        # 2. Build prompt and call LLM (with route-aware budgets)
        phase = self._detect_conversation_phase()
        prompt = self._build_conversation_prompt(user_message, phase, route=route)

        client = get_client()
        system_prompt = CONVERSATION_SYSTEM_PROMPT.format(
            today=datetime.now().date().isoformat(),
            year=datetime.now().year,
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_truncate(
                    system_prompt,
                    TOKEN_BUDGETS[phase]["system"],
                ),
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=CONVERSATION_RESPONSE_SCHEMA,
            ),
        )

        # 3. Parse structured response
        try:
            raw_json = extract_json(response.text)
            result = _normalize_llm_response(raw_json)
        except ValueError:
            # Fallback: treat entire response as text
            result = {
                "response_text": response.text.strip(),
                "extracted_beliefs": [],
                "structured_core_updates": {},
                "trigger_cycle": False,
                "cycle_reason": None,
                "onboarding_complete": False,
            }

        response_text = result.get("response_text", "I'm here to help with your training.")

        # 4. Process extracted beliefs through Mem0 compare/merge
        candidates = result.get("extracted_beliefs", [])
        if candidates:
            self._process_belief_candidates(candidates)

        # 5. Apply structured_core updates
        core_updates = result.get("structured_core_updates", {})
        for field_path, value in core_updates.items():
            if value is not None:
                self.user_model.update_structured_core(field_path, value)

        # 6. Handle cycle trigger
        if result.get("trigger_cycle"):
            self._cycle_triggered = True

        # 6.3 Refresh proactive triggers periodically (P8 — audit finding #7)
        self._maybe_refresh_proactive_triggers()

        # 6.5 Mid-conversation proactive injection (P8 — audit finding #7)
        proactive_note = self._check_proactive_injection(user_message, route)
        if proactive_note:
            response_text = response_text.rstrip() + "\n\n" + proactive_note

        # 7. Update rolling summary if needed
        if self._turns_since_summary >= ROLLING_SUMMARY_INTERVAL:
            self._update_rolling_summary()

        # 8. Log agent response with route metadata
        artifacts = []
        if result.get("trigger_cycle"):
            artifacts.append(f"cycle_triggered:{result.get('cycle_reason', 'unknown')}")
        self._append_to_session("agent", response_text, metadata={"artifacts": artifacts, "route": route})
        self._turn_count += 1

        # 9. Auto-save user model periodically
        self.user_model.meta["last_interaction"] = _now_iso()
        self.user_model.save()

        return response_text

    # ── 5-Level Context Builder ──────────────────────────────────

    def _build_conversation_prompt(self, user_message: str, phase: str = "ongoing", route: str | None = None) -> str:
        """Build the full 5-level context prompt for the LLM.

        Args:
            user_message: The user's message text
            phase: Conversation phase (onboarding, early, ongoing, planning)
            route: Optional message route type from router for budget optimization
        """
        budgets = TOKEN_BUDGETS.get(phase, TOKEN_BUDGETS["ongoing"])

        # Apply route-specific budget overrides if available
        if route:
            try:
                from src.agent.router import get_budget_overrides
                budgets = get_budget_overrides(route, budgets)
            except Exception:
                pass
        parts = []

        # Level 2: User Model Summary
        model_summary = self.user_model.get_model_summary()
        if model_summary:
            parts.append(
                f"=== ATHLETE PROFILE & COACH'S NOTES ===\n"
                f"{_truncate(model_summary, budgets['model'])}\n"
            )

        # Load plan early so both activity context and plan section can use it
        current_plan = self._load_current_plan()

        # Level 2.3: Training Data (activity context)
        if budgets.get("activity", 0) > 0:
            from src.tools.activity_context import build_activity_context
            activity_ctx = build_activity_context(plan=current_plan)
            if activity_ctx:
                parts.append(
                    f"=== TRAINING DATA ===\n"
                    f"{_truncate(activity_ctx, budgets['activity'])}\n"
                )

        # Level 2.5: Current Training Plan (if available)
        if current_plan:
            plan_text = self._format_plan_for_context(current_plan)
            parts.append(
                f"=== CURRENT TRAINING PLAN ===\n"
                f"{_truncate(plan_text, 3000)}\n"
            )

        # Level 3: Cross-Session Context (last 2-3 session summaries)
        if budgets["cross"] > 0:
            cross_context = self._load_session_summaries(n=3)
            if cross_context:
                parts.append(
                    f"=== PREVIOUS SESSION CONTEXT ===\n"
                    f"{_truncate(cross_context, budgets['cross'])}\n"
                )

        # Level 4: Rolling Summary (older turns from this session)
        if self._rolling_summary and budgets["rolling"] > 0:
            parts.append(
                f"=== EARLIER IN THIS SESSION ===\n"
                f"{_truncate(self._rolling_summary, budgets['rolling'])}\n"
            )

        # Level 5: Recent Turns (last 8-10 verbatim)
        recent = self._get_recent_turns_text(RECENT_TURNS_COUNT)
        if recent:
            parts.append(
                f"=== RECENT CONVERSATION ===\n"
                f"{_truncate(recent, budgets['recent'])}\n"
            )

        # Current message
        parts.append(f"=== CURRENT MESSAGE ===\nUser: {user_message}")

        return "\n\n".join(parts)

    def _maybe_refresh_proactive_triggers(self) -> None:
        """Refresh the proactive queue with new triggers from current data.

        P8 enhancement: periodically scans activities, episodes, and trajectory
        during conversation to detect new proactive triggers. Runs every
        PROACTIVE_REFRESH_INTERVAL turns to avoid excessive computation.
        """
        from src.agent.proactive import PROACTIVE_REFRESH_INTERVAL

        # Only refresh every N user turns (turn_count includes agent turns,
        # so divide by 2 to approximate user turn count)
        user_turns = self._turn_count // 2
        if user_turns < 1 or user_turns % PROACTIVE_REFRESH_INTERVAL != 0:
            return

        try:
            from src.agent.proactive import refresh_proactive_triggers
            from src.tools.activity_store import list_activities
            from src.memory.episodes import list_episodes
            from src.agent.trajectory import assess_trajectory

            profile = self.user_model.structured_core
            activities = list_activities()
            episodes = list_episodes(limit=10)

            # Lightweight trajectory for trigger detection
            trajectory = {}
            try:
                trajectory = assess_trajectory(
                    profile, activities, episodes,
                    self._load_current_plan() or {"sessions": []},
                )
            except Exception:
                pass

            refresh_proactive_triggers(
                activities=activities,
                episodes=episodes,
                trajectory=trajectory,
                athlete_profile=profile,
                context={"goal": profile.get("goal", {})},
            )
        except Exception:
            pass  # Never fail the conversation due to proactive refresh

    def _check_proactive_injection(self, user_message: str, route: str | None) -> str | None:
        """Check if a proactive insight should be injected into the response.

        P8 enhancement: mid-conversation proactive injection. Uses LLM to
        decide if any pending proactive messages are relevant to the current
        conversation topic. Skips injection for greetings/thanks (general_chat).

        Returns the proactive note text, or None.
        """
        # Don't inject on simple chat (greetings, thanks)
        if route == "general_chat":
            return None

        try:
            from src.agent.proactive import get_pending_messages, deliver_message

            pending = get_pending_messages()
            if not pending:
                return None

            # Build message descriptions from stored message_text or fallback
            def _describe(m: dict) -> str:
                text = m.get("message_text", "")
                if not text:
                    # Backwards compat: format from trigger_type + data
                    data = m.get("data", {})
                    text = data.get("message", "") or data.get("reasoning", "") or str(data)
                return f"[{m.get('trigger_type', '?')}] {text}"

            messages_summary = "\n".join(
                f"- {_describe(m)}" for m in pending[:3]
            )

            prompt = (
                "You are a coaching AI. Decide if any of these insights should be "
                "naturally mentioned given the athlete's current message.\n\n"
                f"Athlete said: \"{user_message[:300]}\"\n\n"
                f"Available insights:\n{messages_summary}\n\n"
                "Rules:\n"
                "- Only inject if genuinely relevant to what they're discussing\n"
                "- Return the insight as a brief, natural coaching note\n"
                "- If nothing is relevant, return empty text\n\n"
                "Respond with ONLY a JSON object:\n"
                '{{"inject": true|false, "index": 0, "note": "brief coaching note"}}'
            )

            client = get_client()
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    genai.types.Content(
                        role="user",
                        parts=[genai.types.Part(text=prompt)],
                    ),
                ],
                config=genai.types.GenerateContentConfig(temperature=0.3),
            )

            result = extract_json(response.text.strip())
            if result.get("inject") and isinstance(result.get("index"), int):
                idx = result["index"]
                if 0 <= idx < len(pending):
                    deliver_message(pending[idx]["id"])
                    return result.get("note", "")
        except Exception:
            pass

        return None

    def _detect_conversation_phase(self) -> str:
        """Determine conversation phase for dynamic token budget allocation."""
        core = self.user_model.structured_core

        # Check if onboarding
        has_sports = bool(core.get("sports"))
        has_event = bool(core.get("goal", {}).get("event"))
        has_days = bool(core.get("constraints", {}).get("training_days_per_week"))

        if not (has_sports and has_event and has_days):
            return "onboarding"

        # Check if planning mode
        if self._cycle_triggered:
            return "planning"

        # Check session count
        sessions_completed = self.user_model.meta.get("sessions_completed", 0)
        if sessions_completed < 5:
            return "early"

        return "ongoing"

    # ── Belief Compare/Merge (Mem0 Two-Phase Pattern) ────────────

    def _process_belief_candidates(self, candidates: list[dict]) -> None:
        """Process extracted belief candidates through the Mem0 two-phase pipeline.

        Phase 1: Embedding-based retrieval (find similar existing beliefs)
        Phase 2: LLM-based decision (ADD/UPDATE/DELETE/NOOP)

        Fallback: For < 10 total beliefs, skip embedding and send all to LLM.
        """
        active_beliefs = self.user_model.get_active_beliefs()
        use_fallback = len(active_beliefs) < 10

        for candidate in candidates:
            text = candidate.get("text", "")
            if not text:
                continue

            if use_fallback:
                # Small belief set: send all to LLM for comparison
                similar = [(b, 1.0) for b in active_beliefs]
            else:
                # Phase 1: Embedding retrieval
                similar = self.user_model.find_similar_beliefs(text, top_k=3)

            # Phase 2: LLM decision
            operation = self._get_belief_operation(candidate, similar)

            op_type = operation.get("operation", "ADD")
            target_id = operation.get("target_belief_id")

            if op_type == "ADD":
                new_belief = self.user_model.add_belief(
                    text=text,
                    category=candidate.get("category", "preference"),
                    confidence=candidate.get("confidence", 0.7),
                    source="conversation",
                    source_ref=self._session_id,
                )
                # Generate embedding for new belief
                self.user_model.embed_belief(new_belief)

            elif op_type == "UPDATE" and target_id:
                new_text = operation.get("new_text", text)
                new_conf = operation.get("new_confidence", candidate.get("confidence", 0.7))
                updated = self.user_model.update_belief(
                    target_id,
                    new_text=new_text,
                    new_confidence=new_conf,
                )
                # Re-generate embedding after text update
                if updated:
                    self.user_model.embed_belief(updated)

            elif op_type == "DELETE" and target_id:
                # Add new belief first, then invalidate old
                new_belief = self.user_model.add_belief(
                    text=text,
                    category=candidate.get("category", "preference"),
                    confidence=candidate.get("confidence", 0.7),
                    source="conversation",
                    source_ref=self._session_id,
                )
                self.user_model.embed_belief(new_belief)
                self.user_model.invalidate_belief(target_id, superseded_by=new_belief["id"])

            # NOOP: do nothing

    def _get_belief_operation(
        self,
        candidate: dict,
        similar_beliefs: list[tuple[dict, float]],
    ) -> dict:
        """Call LLM to decide belief operation (ADD/UPDATE/DELETE/NOOP).

        For very few beliefs or when all are low similarity, defaults to ADD.
        """
        if not similar_beliefs:
            return {"operation": "ADD", "target_belief_id": None}

        # Build context for LLM
        existing_text = ""
        for belief, score in similar_beliefs:
            existing_text += (
                f"- ID: {belief['id']}\n"
                f"  Text: {belief['text']}\n"
                f"  Category: {belief['category']}\n"
                f"  Confidence: {belief['confidence']}\n"
                f"  Similarity: {score:.2f}\n\n"
            )

        prompt = (
            f"CANDIDATE BELIEF:\n"
            f"  Text: {candidate['text']}\n"
            f"  Category: {candidate.get('category', 'unknown')}\n"
            f"  Confidence: {candidate.get('confidence', 0.7)}\n\n"
            f"EXISTING BELIEFS:\n{existing_text}\n"
            f"What operation should be performed?"
        )

        try:
            client = get_client()
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=BELIEF_MERGE_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            return extract_json(response.text)
        except Exception:
            # On any failure, default to ADD
            return {"operation": "ADD", "target_belief_id": None}

    # ── Session Storage (JSONL) ──────────────────────────────────

    def _append_to_session(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Append a turn to the current session JSONL file."""
        if not self._session_file:
            return

        self._session_file.parent.mkdir(parents=True, exist_ok=True)

        turn = {
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
            "artifacts": (metadata or {}).get("artifacts", []),
        }

        with open(self._session_file, "a") as f:
            f.write(json.dumps(turn) + "\n")

    def get_recent_turns(self, n: int = 10) -> list[dict]:
        """Return last n turns from current session JSONL."""
        if not self._session_file or not self._session_file.exists():
            return []

        turns = []
        with open(self._session_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return turns[-n:]

    def _get_recent_turns_text(self, n: int = 10) -> str:
        """Format recent turns as text for prompt injection."""
        turns = self.get_recent_turns(n)
        lines = []
        for t in turns:
            role = "Coach" if t["role"] == "agent" else "Athlete"
            lines.append(f"{role}: {t['content']}")
        return "\n".join(lines)

    # ── Rolling Summary ──────────────────────────────────────────

    def _get_rolling_summary(self) -> str:
        """Return the current rolling summary."""
        return self._rolling_summary

    def _update_rolling_summary(self) -> None:
        """Regenerate rolling summary via LLM (full re-summarization).

        Summarizes all turns BEFORE the most recent RECENT_TURNS_COUNT turns.
        """
        if not self._session_file or not self._session_file.exists():
            return

        all_turns = []
        with open(self._session_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Only summarize turns before the recent window
        older_turns = all_turns[:-RECENT_TURNS_COUNT] if len(all_turns) > RECENT_TURNS_COUNT else []
        if not older_turns:
            self._turns_since_summary = 0
            return

        # Build conversation text for summarization
        conversation_text = ""
        for t in older_turns:
            role = "Coach" if t["role"] == "agent" else "Athlete"
            conversation_text += f"{role}: {t['content']}\n"

        try:
            client = get_client()
            response = client.models.generate_content(
                model=MODEL,
                contents=f"Conversation to summarize:\n\n{conversation_text}",
                config=genai.types.GenerateContentConfig(
                    system_instruction=ROLLING_SUMMARY_PROMPT,
                    temperature=0.3,
                ),
            )
            self._rolling_summary = response.text.strip()
        except Exception:
            pass  # Keep existing summary on failure

        self._turns_since_summary = 0

    # ── Session Summaries (Cross-Session) ────────────────────────

    def _load_session_summaries(self, n: int = 3) -> str:
        """Load last n session summaries and format for prompt injection."""
        summary_files = sorted(
            self._sessions_dir.glob("summary_*.json"),
            reverse=True,
        )

        summaries = []
        for path in summary_files[:n]:
            try:
                data = json.loads(path.read_text())
                text = (
                    f"Session {data.get('date', 'unknown')} "
                    f"({data.get('duration_turns', '?')} turns):\n"
                    f"{data.get('summary', 'No summary available.')}\n"
                    f"Topics: {', '.join(data.get('key_topics', []))}\n"
                    f"Next time: {data.get('next_session_context', 'N/A')}"
                )
                summaries.append(text)
            except (json.JSONDecodeError, OSError):
                continue

        return "\n\n".join(summaries)

    def _generate_session_summary(self) -> dict:
        """Generate a structured session summary via LLM."""
        all_turns = self.get_recent_turns(n=9999)  # all turns
        if not all_turns:
            return {
                "session_id": self._session_id,
                "date": datetime.now().date().isoformat(),
                "duration_turns": 0,
                "summary": "Empty session.",
                "key_topics": [],
                "decisions_made": [],
                "athlete_mood": "unknown",
                "next_session_context": "",
            }

        conversation_text = ""
        for t in all_turns:
            role = "Coach" if t["role"] == "agent" else "Athlete"
            conversation_text += f"{role}: {t['content']}\n"

        client = get_client()
        response = client.models.generate_content(
            model=MODEL,
            contents=f"Session to summarize:\n\n{conversation_text}",
            config=genai.types.GenerateContentConfig(
                system_instruction=SESSION_SUMMARY_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        try:
            summary_data = extract_json(response.text)
        except ValueError:
            summary_data = {
                "summary": response.text.strip(),
                "key_topics": [],
                "decisions_made": [],
                "athlete_mood": "unknown",
                "next_session_context": "",
            }

        # Enrich with metadata
        summary_data["session_id"] = self._session_id
        summary_data["date"] = datetime.now().date().isoformat()
        summary_data["duration_turns"] = self._turn_count
        summary_data["beliefs_extracted_count"] = len(
            [b for b in self.user_model.beliefs if b.get("source_ref") == self._session_id]
        )

        return summary_data

    # ── On-Demand Session Search (BM25) ──────────────────────────

    def search_sessions(self, query: str, top_k: int = 3) -> list[dict]:
        """Search past session content using BM25.

        Searches over: session summaries + session transcripts + plan files.
        Returns relevant context chunks with source references.
        """
        if self._bm25_index is None:
            self._build_bm25_index()

        if self._bm25_index is None or not self._bm25_corpus:
            return []

        try:
            import bm25s
            # Tokenize query
            query_tokens = bm25s.tokenize([query], show_progress=False)
            results, scores = self._bm25_index.retrieve(
                query_tokens, k=min(top_k, len(self._bm25_corpus))
            )

            hits = []
            for i in range(results.shape[1]):
                idx = int(results[0, i])
                score = float(scores[0, i])
                if score > 0 and idx < len(self._bm25_corpus):
                    hit = self._bm25_corpus[idx].copy()
                    hit["score"] = score
                    hits.append(hit)

            return hits[:top_k]
        except Exception:
            return []

    def _build_bm25_index(self) -> None:
        """Build BM25 index over session summaries, transcripts, and plans."""
        try:
            import bm25s
        except ImportError:
            return

        corpus = []
        texts = []

        # Index session summaries
        for path in self._sessions_dir.glob("summary_*.json"):
            try:
                data = json.loads(path.read_text())
                text = data.get("summary", "")
                if text:
                    corpus.append({
                        "type": "summary",
                        "source": path.name,
                        "text": text,
                    })
                    texts.append(text)
            except (json.JSONDecodeError, OSError):
                continue

        # Index session transcripts
        for path in self._sessions_dir.glob("session_*.jsonl"):
            try:
                session_text = ""
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            turn = json.loads(line)
                            session_text += f"{turn['role']}: {turn['content']}\n"
                if session_text:
                    corpus.append({
                        "type": "transcript",
                        "source": path.name,
                        "text": session_text[:2000],  # limit size
                    })
                    texts.append(session_text[:2000])
            except (json.JSONDecodeError, OSError):
                continue

        # Index plan files
        plans_dir = self._data_dir / "plans"
        if plans_dir.exists():
            for path in sorted(plans_dir.glob("plan_*.json"), reverse=True)[:5]:
                try:
                    data = json.loads(path.read_text())
                    summary = data.get("weekly_summary", {})
                    text = f"Plan {path.stem}: {summary.get('focus', '')} - {summary.get('total_sessions', '?')} sessions"
                    corpus.append({
                        "type": "plan",
                        "source": path.name,
                        "text": text,
                    })
                    texts.append(text)
                except (json.JSONDecodeError, OSError):
                    continue

        if not texts:
            return

        self._bm25_corpus = corpus
        corpus_tokens = bm25s.tokenize(texts, show_progress=False)
        self._bm25_index = bm25s.BM25()
        self._bm25_index.index(corpus_tokens)

    @staticmethod
    def _format_targets_text(targets: dict) -> str:
        """Format sport-specific targets as compact text for LLM context.

        Similar to cli.py's _format_targets but uses space separators
        (not pipes) since this is for LLM consumption, not human display.

        Running: "5:30-6:00/km Zone 2"
        Cycling: "180-200W Zone 3-4 85-95rpm"
        Swimming: "1:45-1:55/100m Zone 2-3"
        General: "Zone 2-3 RPE 6-7"
        """
        if not targets:
            return ""

        parts = []
        if "pace_min_km" in targets:
            parts.append(f"{targets['pace_min_km']}/km")
        if "power_watts" in targets:
            parts.append(f"{targets['power_watts']}W")
        if "pace_min_100m" in targets:
            parts.append(f"{targets['pace_min_100m']}/100m")
        if "cadence_rpm" in targets:
            parts.append(f"{targets['cadence_rpm']}rpm")
        if "hr_zone" in targets:
            parts.append(targets["hr_zone"])
        if "rpe" in targets:
            parts.append(f"RPE {targets['rpe']}")

        return " ".join(parts)

    # ── Plan Context ─────────────────────────────────────────────

    def _load_current_plan(self) -> dict | None:
        """Load the most recent training plan from data/plans/."""
        plans_dir = self._data_dir / "plans"
        if not plans_dir.exists():
            return None
        plan_files = sorted(plans_dir.glob("plan_*.json"))
        if not plan_files:
            return None
        try:
            return json.loads(plan_files[-1].read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _format_plan_for_context(self, plan: dict) -> str:
        """Format a training plan as concise text for prompt injection.

        Handles both old flat-format plans (target_pace_min_km, description)
        and new structured-steps plans (steps array with repeat groups).
        """
        lines = []
        week_start = plan.get("week_start", "?")
        lines.append(f"Week starting {week_start}:")

        for s in plan.get("sessions", []):
            day = s.get("day", "?")
            sport = s.get("sport", "?")
            stype = s.get("type", "?")

            if s.get("steps"):
                # New format: structured workout steps
                dur = s.get(
                    "total_duration_minutes",
                    s.get("duration_minutes", "?"),
                )
                lines.append(f"  {day}: {sport} - {stype} ({dur}min)")

                for step in s["steps"]:
                    step_type = step.get("type", "?")

                    if step_type == "repeat":
                        count = step.get("repeat_count", 1)
                        lines.append(f"    {count}x:")
                        for sub in step.get("steps", []):
                            sub_type = sub.get("type", "?").title()
                            sub_dur = sub.get("duration", "")
                            sub_targets = self._format_targets_text(
                                sub.get("targets", {})
                            )
                            at_str = f" @ {sub_targets}" if sub_targets else ""
                            lines.append(f"      {sub_type} {sub_dur}{at_str}")
                    else:
                        st = step_type.title()
                        dur_s = step.get("duration", "")
                        targets_text = self._format_targets_text(
                            step.get("targets", {})
                        )
                        at_str = f" @ {targets_text}" if targets_text else ""
                        lines.append(f"    {st} {dur_s}{at_str}")

                notes = s.get("notes", "")
                if notes:
                    lines.append(f"    Note: {notes}")
            else:
                # Old flat format: preserve existing behavior exactly
                dur = s.get("duration_minutes", "?")
                pace = s.get("target_pace_min_km", "")
                hr = s.get("target_hr_zone", "")
                desc = s.get("description", "")
                target = f" @ {pace}" if pace else (f" {hr}" if hr else "")
                lines.append(f"  {day}: {sport} - {stype} ({dur}min){target}")
                if desc:
                    lines.append(f"    {desc}")

        summary = plan.get("weekly_summary", {})
        if summary:
            lines.append(
                f"  Summary: {summary.get('total_sessions', '?')} sessions, "
                f"{summary.get('total_duration_minutes', '?')}min total, "
                f"focus: {summary.get('focus', '?')}"
            )

        return "\n".join(lines)

    # ── Properties ───────────────────────────────────────────────

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def cycle_triggered(self) -> bool:
        return self._cycle_triggered

    def reset_cycle_trigger(self) -> None:
        """Reset the cycle trigger flag after processing."""
        self._cycle_triggered = False
