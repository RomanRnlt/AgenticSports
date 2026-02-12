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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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

        # 2. Build prompt and call LLM
        phase = self._detect_conversation_phase()
        prompt = self._build_conversation_prompt(user_message, phase)

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
            ),
        )

        # 3. Parse structured response
        try:
            result = extract_json(response.text)
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

        # 7. Update rolling summary if needed
        if self._turns_since_summary >= ROLLING_SUMMARY_INTERVAL:
            self._update_rolling_summary()

        # 8. Log agent response
        artifacts = []
        if result.get("trigger_cycle"):
            artifacts.append(f"cycle_triggered:{result.get('cycle_reason', 'unknown')}")
        self._append_to_session("agent", response_text, metadata={"artifacts": artifacts})
        self._turn_count += 1

        # 9. Auto-save user model periodically
        self.user_model.meta["last_interaction"] = _now_iso()
        self.user_model.save()

        return response_text

    # ── 5-Level Context Builder ──────────────────────────────────

    def _build_conversation_prompt(self, user_message: str, phase: str = "ongoing") -> str:
        """Build the full 5-level context prompt for the LLM."""
        budgets = TOKEN_BUDGETS.get(phase, TOKEN_BUDGETS["ongoing"])
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
        """Format a training plan as concise text for prompt injection."""
        lines = []
        week_start = plan.get("week_start", "?")
        lines.append(f"Week starting {week_start}:")

        for s in plan.get("sessions", []):
            day = s.get("day", "?")
            sport = s.get("sport", "?")
            stype = s.get("type", "?")
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
