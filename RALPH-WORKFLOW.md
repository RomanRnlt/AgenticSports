# RALPH for Autonomous Agent Development: A Q1 2026 Deep Analysis

## Applied to the ReAgt Project

*February 2026 | Workflow Design & Best Practices*

---

## Table of Contents

1. [What RALPH Actually Is (and Isn't)](#1-what-ralph-actually-is-and-isnt)
2. [RALPH's Core Mechanisms](#2-ralphs-core-mechanisms)
3. [How RALPH Manages Autonomy, State, and Progress](#3-how-ralph-manages-autonomy-state-and-progress)
4. [Applying RALPH to the ReAgt Project](#4-applying-ralph-to-the-reagt-project)
5. [Q1 2026 Best-Practice Workflow for RALPH](#5-q1-2026-best-practice-workflow-for-ralph)
6. [Agent & State Management in Detail](#6-agent--state-management-in-detail)
7. [Git as a Core Mechanism](#7-git-as-a-core-mechanism)
8. [Alternatives & Comparison](#8-alternatives--comparison)
9. [Common Mistakes When Using RALPH (and How to Avoid Them)](#9-common-mistakes-when-using-ralph-and-how-to-avoid-them)
10. [Summary: The ReAgt + RALPH Execution Plan](#10-summary-the-reagt--ralph-execution-plan)

---

## 1. What RALPH Actually Is (and Isn't)

RALPH is **not a framework, SDK, or agent system**. It is a **supervisor shell script** (~2,000 lines of Bash) that wraps Claude Code in an instrumented loop. The core insight comes from Geoffrey Huntley's original "Ralph Wiggum" technique:

```bash
while true; do claude -p "$(cat PROMPT.md)"; done
```

Claude Code operates on a persistent filesystem. When invoked repeatedly, it sees its own previous work on disk, identifies what's broken or incomplete, and continues. Over many iterations, a complete project emerges.

What `frankbria/ralph-claude-code` adds to the raw loop:

| Concern | Raw Loop | RALPH |
|---|---|---|
| When to stop | Never (manual Ctrl+C) | Dual-condition exit gate + 6 exit criteria |
| Stagnation detection | None | Three-state circuit breaker (CLOSED -> HALF_OPEN -> OPEN) |
| Rate limiting | None | Configurable calls/hour with countdown |
| Session continuity | None | `--resume <session-id>` with 24h expiry |
| Progress tracking | None | Git SHA comparison before/after each loop |
| Monitoring | `tail -f` | tmux dashboard with live streaming |
| Configuration | Edit the script | `.ralphrc` file sourced as Bash |
| Project setup | Manual | Interactive wizard, PRD import, or scaffold |

**Critical framing**: RALPH never reads, evaluates, or understands the code Claude writes. It analyzes Claude's *output text* for structured status blocks. All code quality, architectural decisions, and testing happen inside Claude Code's invocation. RALPH decides only: **continue, wait, or stop**.

---

## 2. RALPH's Core Mechanisms

### 2.1 The Loop Iteration as Atomic Unit

Each loop iteration is one Claude Code invocation. The command RALPH builds looks like:

```bash
npx @anthropic-ai/claude-code \
  --output-format json \
  --allowedTools "Write,Read,Edit,Bash(git *),Bash(npm *),Bash(pytest)" \
  --resume "session-uuid" \
  --append-system-prompt "RALPH LOOP CONTEXT: Loop #7. Remaining tasks: 5. Previous: implemented auth module. Circuit breaker: CLOSED." \
  -p "$(cat .ralph/PROMPT.md)"
```

Each iteration is expected to:

1. Read `.ralph/fix_plan.md` for current priorities
2. Read `.ralph/specs/` for detailed requirements
3. Pick the highest-priority unchecked task
4. Implement it (code + tests)
5. Update `fix_plan.md` to mark it complete
6. Commit with a descriptive message
7. Output a structured `RALPH_STATUS` block

### 2.2 The RALPH_STATUS Contract

This is the structured output Claude **must** produce each iteration:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary>
---END_RALPH_STATUS---
```

The `PROMPT.md` template includes 6 Specification-by-Example scenarios teaching Claude exactly when to output which values. The EXIT_SIGNAL rules are explicit: set to `true` only when ALL tasks in fix_plan.md are `[x]`, all tests pass, no errors remain, and nothing meaningful is left.

### 2.3 Exit Detection (The Dual-Condition Gate)

RALPH's most important design decision: it requires **two independent signals** to stop.

| completion_indicators (>=2) | EXIT_SIGNAL (latest) | Result |
|---|---|---|
| Yes | `true` | **EXIT** -- project_complete |
| Yes | `false` | Continue (Claude still working) |
| No | `true` | Continue (threshold not met yet) |
| No | `false` | Continue |

This prevents premature exit when Claude says "Phase 1 complete, moving to Phase 2." Additional exit triggers, checked in priority order:

1. Permission denial -> immediate exit
2. Test saturation (3 consecutive test-only loops)
3. Consecutive done signals (>=2)
4. Safety cap (5 consecutive completion indicators)
5. Plan complete (all checkboxes `[x]` in fix_plan.md)

### 2.4 The Circuit Breaker

A three-state system detecting stagnation:

```
CLOSED --(2 loops no progress)--> HALF_OPEN --(threshold met)--> OPEN
   ^                                  |                            |
   |                                  |                            |
   +------(progress detected)---------+    (cooldown/auto-reset)---+
```

"Progress" is detected from three sources (any one suffices):

- Git SHA changed (commits were made)
- Claude reported `STATUS: COMPLETE`
- Claude self-reported `files_modified > 0`

Configurable thresholds (via `.ralphrc`):

- `CB_NO_PROGRESS_THRESHOLD=3` -- loops without file changes before OPEN
- `CB_SAME_ERROR_THRESHOLD=5` -- repeated identical errors before OPEN
- `CB_OUTPUT_DECLINE_THRESHOLD=70` -- % output decline before OPEN
- `CB_COOLDOWN_MINUTES=30` -- auto-recovery time
- `CB_AUTO_RESET=false` -- reset to CLOSED on startup

### 2.5 State Management: Everything Is Files

All state lives in `.ralph/` as flat files:

| File | Purpose |
|---|---|
| `.ralph/.call_count` | API calls this hour |
| `.ralph/.last_reset` | Hourly counter reset timestamp |
| `.ralph/.exit_signals` | Rolling window (last 5) of signal-triggering loops |
| `.ralph/.response_analysis` | Latest Claude output analysis (JSON) |
| `.ralph/.circuit_breaker_state` | CB state, counters, timestamps (JSON) |
| `.ralph/.claude_session_id` | Session UUID for `--resume` |
| `.ralph/.loop_start_sha` | Git HEAD before iteration (for diff) |
| `.ralph/status.json` | Machine-readable status for monitoring |
| `.ralph/logs/ralph.log` | Execution log |
| `.ralph/logs/claude_output_*.log` | Raw output per iteration |

---

## 3. How RALPH Manages Autonomy, State, and Progress

### Autonomy Model

RALPH operates at what the community calls **"sit on the loop, not in it"** autonomy. There is no human-in-the-loop checkpoint during execution. The only intervention points are:

1. **Before starting**: Human writes PROMPT.md, fix_plan.md, specs/, .ralphrc
2. **During execution**: Human watches the tmux monitor (optional)
3. **On circuit breaker OPEN**: Human diagnoses why progress stopped
4. **After completion**: Human reviews the result

There is no "approve before proceeding" gate within RALPH itself. If you need that, you build it into your prompt (e.g., "After completing each step, write a REVIEW_NEEDED.md file and set STATUS: BLOCKED").

### State Continuity Across Sessions

RALPH preserves state through two mechanisms:

**1. Session continuity (`--resume`)**: Claude Code maintains its conversation context across loop iterations within a session. The session expires after 24 hours (configurable via `SESSION_EXPIRY_HOURS`). Within a session, Claude remembers what it did in previous iterations.

**2. Filesystem state (always persisted)**: Even after session expiry, all project state survives on disk:

- The code Claude wrote
- Git history of all commits
- `fix_plan.md` with checked/unchecked tasks
- `specs/` with requirements
- Any logs or artifacts

On restart after interruption:

- Rate limits: Checked against timestamp, reset if hour elapsed
- Circuit breaker: Auto-reset or cooldown recovery
- Session: Resume if valid, fresh start if expired
- Progress: Claude reads fix_plan.md and continues from unchecked tasks

### Progress Tracking

Progress is fundamentally tracked through **two parallel systems**:

1. **Git diffs**: Before each iteration, RALPH records `git rev-parse HEAD`. After, it compares. If the SHA changed, files were committed -- that's progress.

2. **fix_plan.md checkboxes**: The prompt instructs Claude to mark completed tasks with `[x]`. RALPH counts checked vs unchecked items to determine completion percentage and detect plan completion.

These are independent signals. Git diffs detect implementation progress. Checkbox counting detects task-level completion. Together, they provide robust tracking without RALPH ever needing to understand the code.

---

## 4. Applying RALPH to the ReAgt Project

### 4.1 Project Structure for RALPH

The ReAgt project needs this structure:

```
reagt/
+-- .ralph/
|   +-- PROMPT.md          # Master instructions for Claude
|   +-- fix_plan.md        # Current increment's task checklist
|   +-- AGENT.md           # Build/test commands
|   +-- specs/
|       +-- architecture.md    # From REAGT.md Section 2
|       +-- step0_setup.md     # From REAGT.md Step 0
|       +-- step1_dumb_coach.md
|       +-- step2_perceiving.md
|       +-- step3_adaptive.md
|       +-- step4_reflecting.md
|       +-- step5_proactive.md
+-- .ralphrc               # RALPH configuration
+-- src/
|   +-- agent/
|   +-- tools/
|   +-- memory/
|   +-- interface/
+-- data/
+-- tests/
+-- pyproject.toml
+-- .env                   # API keys (gitignored)
```

### 4.2 Mapping Roadmap Steps to RALPH Increments

Each step in REAGT.md becomes one RALPH "run" -- a complete loop session from start to exit.

#### Increment 0: Project Setup

**`.ralph/fix_plan.md`:**

```markdown
# Step 0: Project Setup

## High Priority
- [ ] Initialize Python project with pyproject.toml (dependencies: google-genai, fitdecode, rich, watchdog)
- [ ] Create directory structure: src/agent, src/tools, src/memory, src/interface, data/fit_files, data/athlete, data/plans, data/episodes, tests/
- [ ] Set up .env loading for GEMINI_API_KEY
- [ ] Verify Gemini API connectivity with a test call
- [ ] Create initial test: test that API key loads and Gemini returns a response
- [ ] Initialize git repository and create .gitignore
```

**`.ralph/specs/step0_setup.md`:** Extract the Step 0 section from REAGT.md verbatim, plus explicit acceptance criteria:

```markdown
## Acceptance Criteria
- `python -m pytest tests/` passes with at least 1 test
- `python -c "from src.agent import core"` does not error
- A call to Gemini 2.0 Flash returns a non-empty response
- .env is in .gitignore
- All directories exist
```

**`.ralphrc`:**

```bash
PROJECT_NAME="reagt"
PROJECT_TYPE="python"
MAX_CALLS_PER_HOUR=50
CLAUDE_TIMEOUT_MINUTES=15
ALLOWED_TOOLS="Write,Read,Edit,Bash(git *),Bash(python *),Bash(pip *),Bash(pytest *),Bash(uv *)"
SESSION_CONTINUITY=true
CB_NO_PROGRESS_THRESHOLD=3
CB_AUTO_RESET=true
```

**How to run it:**

```bash
cd reagt
ralph --monitor
```

RALPH will loop until fix_plan.md has all `[x]` and EXIT_SIGNAL fires twice. For Step 0, expect 3-8 iterations.

#### Increment 1: The Dumb Coach

After Step 0 completes, **manually update** `.ralph/fix_plan.md` and add the Step 1 spec:

**`.ralph/fix_plan.md`:**

```markdown
# Step 1: The Dumb Coach

## High Priority
- [ ] Create system prompt for training coach role
- [ ] Implement CLI that asks user for sports, goals, target event
- [ ] Implement athlete_profile.json creation via Gemini function calling
- [ ] Implement 1-week training plan generation with structured JSON output
- [ ] Save plan to data/plans/ with timestamp
- [ ] Display plan in terminal using Rich

## Medium Priority
- [ ] Write tests for profile creation (valid input -> valid JSON)
- [ ] Write tests for plan generation (profile -> structured plan)
- [ ] Handle Gemini API errors gracefully
```

**Run again:**

```bash
ralph --monitor
```

#### Increments 2-5: Same Pattern

Each step follows the same cycle:

1. Human updates `fix_plan.md` with the new step's tasks
2. Human adds/updates the relevant spec in `specs/`
3. Run `ralph --monitor`
4. RALPH loops until completion
5. Human reviews the result, then prepares the next increment

### 4.3 The PROMPT.md for ReAgt

This is the single most important file:

```markdown
You are Ralph, an autonomous AI development agent building ReAgt --
an adaptive training agent for endurance athletes.

## Project Context
- Python project using Gemini 2.0 Flash as LLM backend
- Architecture: State machine with persistent memory (JSON files)
- See .ralph/specs/architecture.md for full architecture

## Your Work Cycle
1. Read .ralph/fix_plan.md for current priorities
2. Read relevant .ralph/specs/ files for requirements
3. Pick the highest-priority unchecked task
4. Implement it with tests
5. Run tests: `python -m pytest tests/ -v`
6. Update fix_plan.md to mark completed tasks [x]
7. Commit working changes with descriptive messages

## Key Principles
- ONE task per loop -- focus on the most important thing
- Search the codebase before assuming something isn't implemented
- Write tests for new functionality (but limit testing to ~20% of effort)
- Keep it simple -- JSON files for storage, no database
- Use google-genai SDK for all Gemini interactions
- All state in data/ directory (JSON files)
- Handle .env for API keys, never commit secrets

## Testing
- Run: python -m pytest tests/ -v
- LIMIT testing to ~20% of total effort per loop
- PRIORITIZE: Implementation > Documentation > Tests
- Only write tests for NEW functionality you implement

## RALPH_STATUS Output (Required at End of Every Loop)

You MUST output the following block at the end of every response:

    ---RALPH_STATUS---
    STATUS: IN_PROGRESS | COMPLETE | BLOCKED
    TASKS_COMPLETED_THIS_LOOP: <number>
    FILES_MODIFIED: <number>
    TESTS_STATUS: PASSING | FAILING | NOT_RUN
    WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
    EXIT_SIGNAL: false | true
    RECOMMENDATION: <one line summary of what to do next>
    ---END_RALPH_STATUS---

### EXIT_SIGNAL Rules

Set EXIT_SIGNAL to true ONLY when ALL of these conditions are met:
1. All items in fix_plan.md are marked [x]
2. All tests are passing
3. No errors or warnings in the last execution
4. All requirements from specs/ are implemented
5. You have nothing meaningful left to implement

### Specification-by-Example Scenarios

**Scenario 1: Successful Implementation**
Given: fix_plan.md has 5 unchecked items
When: You implement the highest-priority item and tests pass
Then: STATUS: IN_PROGRESS, EXIT_SIGNAL: false, TASKS_COMPLETED_THIS_LOOP: 1

**Scenario 2: All Tasks Complete**
Given: fix_plan.md has 0 unchecked items, all tests passing
When: You verify everything works
Then: STATUS: COMPLETE, EXIT_SIGNAL: true, TASKS_COMPLETED_THIS_LOOP: 0

**Scenario 3: Stuck on Error**
Given: A dependency is missing or an external service is unreachable
When: You cannot proceed after reasonable attempts
Then: STATUS: BLOCKED, EXIT_SIGNAL: false, RECOMMENDATION: <what needs to be fixed>

**Scenario 4: Test-Only Loop**
Given: All implementation is done but some tests need fixing
When: You spend the loop fixing tests
Then: STATUS: IN_PROGRESS, WORK_TYPE: TESTING, EXIT_SIGNAL: false

**Scenario 5: Phase Complete, More Work Ahead**
Given: Current step's tasks are done but next step exists
When: You've completed all items in fix_plan.md
Then: STATUS: COMPLETE, EXIT_SIGNAL: true

**Scenario 6: Making Progress**
Given: fix_plan.md has items, you implemented some
When: Tests pass, code works
Then: STATUS: IN_PROGRESS, EXIT_SIGNAL: false
```

### 4.4 Secrets and Environment Handling

For ReAgt specifically:

**What to do:**

- Create `.env` with `GEMINI_API_KEY=your-key-here`
- Add `.env` to `.gitignore` before first commit
- In your specs, tell Claude to use `python-dotenv` or `os.getenv("GEMINI_API_KEY")`
- In `.ralph/AGENT.md`, document: "Before running, ensure .env exists with GEMINI_API_KEY"

**What NOT to do:**

- Don't put API keys in PROMPT.md, fix_plan.md, or specs
- Don't grant RALPH's `ALLOWED_TOOLS` access to `Bash(env *)` or `Bash(printenv *)`
- Don't rely on Claude to set up the API key -- do this manually before the first run

**In your Step 0 spec, include:**

```markdown
## Environment Setup (Manual -- done before RALPH runs)
- .env file exists with GEMINI_API_KEY
- .env is gitignored
- Code loads API key via: os.environ.get("GEMINI_API_KEY")
```

This is a **manual prerequisite**, not a RALPH task. RALPH should validate that the key works (test call), not create or manage the key.

---

## 5. Q1 2026 Best-Practice Workflow for RALPH

Based on how power users actually operate in January 2026:

### Phase 1: Preparation (Human, ~30 minutes)

1. **Decompose your roadmap into increments** -- each increment = one RALPH run
2. **Write specs** -- one file per increment in `.ralph/specs/`
3. **Write PROMPT.md** -- project-specific, with the standard RALPH_STATUS contract
4. **Write AGENT.md** -- build/test commands Claude needs
5. **Write fix_plan.md** -- checklist for the first increment only
6. **Configure .ralphrc** -- tool permissions, rate limits, circuit breaker thresholds
7. **Set up .env** -- API keys, gitignored
8. **Verify manually** -- `git init`, confirm Claude Code CLI works

### Phase 2: Execution (RALPH, autonomous)

```bash
ralph --monitor    # tmux dashboard with live output
```

**What to expect per increment:**

- Step 0 (setup): 3-8 iterations, ~15-30 minutes
- Step 1 (basic feature): 5-15 iterations, ~30-60 minutes
- Step 2+ (complex features): 10-30 iterations, ~1-3 hours

**What to watch for in the monitor:**

- Circuit breaker transitioning to HALF_OPEN (investigate before it goes OPEN)
- Test saturation (Claude stuck writing tests instead of implementing)
- Repeated errors in output (same failure loop)

### Phase 3: Review (Human, after each increment)

1. `git log --oneline` -- review what Claude committed
2. Run tests manually -- verify they actually pass
3. Read the code -- RALPH cannot judge code quality
4. Check `fix_plan.md` -- confirm all tasks make sense as done
5. Prepare the next increment's `fix_plan.md` and update specs

### Phase 4: Iterate

Update `fix_plan.md` for the next increment. Run `ralph --monitor` again.

### Cost Management

Power users report:

- Small increments (20 iterations): $20-50
- Medium increments (50 iterations): $50-100
- Always set `MAX_CALLS_PER_HOUR` to a sane cap

For ReAgt with 6 steps, expect total cost in the range of $150-400 depending on complexity and iteration count.

---

## 6. Agent & State Management in Detail

### RALPH Is Single-Agent

RALPH runs exactly one Claude Code process at a time. There is no multi-agent orchestration, no supervisor/worker hierarchy. When the prompt says "Use parallel subagents," it refers to Claude Code's *internal* sub-agent capability (Claude spawning child processes for file search), not RALPH managing multiple Claude instances.

### Implicit Role Separation

Though single-agent, RALPH achieves role separation through files:

| File | Role |
|---|---|
| `PROMPT.md` | Product owner (what to build) |
| `fix_plan.md` | Sprint backlog (priority order) |
| `specs/` | Business analyst (detailed requirements) |
| Claude Code | Developer (plans, codes, tests, commits) |
| `response_analyzer.sh` | Quality gate (did anything change?) |
| `circuit_breaker.sh` | Ops monitor (is progress happening?) |
| Exit detection | Release manager (is it done?) |

### State Passing Between Iterations

**Within a session** (same 24h window, same session ID): Claude has full conversation history via `--resume`. It remembers what it did, what failed, what's next.

**Across sessions** (session expired or manually reset): Claude starts fresh but reads the filesystem:

- `fix_plan.md` tells it what's done and what's remaining
- `specs/` tells it what to build
- The codebase itself shows what exists
- Git history shows what was attempted
- `--append-system-prompt` provides ~500 chars of context bridging

**This is the key insight**: RALPH's state is the filesystem itself. The code, the fix_plan, the specs, the git history -- these ARE the state. Claude reads them fresh each session, just as a human developer would read the codebase before starting work.

### Resumption After Interruption

On `Ctrl+C`: RALPH's trap handler resets session state, clears exit signals, writes "stopped" to status.json.

On restart: Rate limits check against timestamp (reset if hour passed). Circuit breaker auto-recovers based on configuration. Session resumes if valid, starts fresh if expired. Claude reads fix_plan.md and continues from unchecked tasks.

**For the ReAgt project**: If RALPH stops mid-increment (circuit breaker, manual stop, API error), simply run `ralph --monitor` again. Claude will read the current state of the code and fix_plan.md and continue. This is robust because each iteration is designed to be idempotent -- Claude looks at what exists and does the next thing.

---

## 7. Git as a Core Mechanism

### Git as Progress Detector

Before each iteration: `git rev-parse HEAD` -> `.ralph/.loop_start_sha`

After each iteration: Compare current HEAD with stored SHA

If they differ -> commits were made -> progress detected -> circuit breaker resets.

This was fixed in v0.11.4 (issue #141). Previously, commits within a loop weren't counted because the code only checked `git diff --name-only HEAD` (which shows zero after a clean commit).

### Git as Historical Record

Every RALPH iteration that produces working code results in a commit (because the prompt instructs Claude to commit). This means:

- `git log` is your progress history
- `git diff step0..step1` shows what each increment produced
- `git bisect` works if a regression appears (though RALPH has no built-in rollback)
- Each commit message (written by Claude) describes what was done

### Git Does NOT Replace Task Tracking

`fix_plan.md` is the task tracker, not git. Git tells you *what code changed*. fix_plan.md tells you *what tasks are done*. RALPH uses both signals independently.

### Branch Strategy

RALPH has **no branch management**. It works on whatever branch is checked out. Recommended approach for ReAgt:

```
main (stable)
  +-- ralph/step-0  (RALPH runs here)
  +-- ralph/step-1  (merge step-0, then RALPH runs here)
  +-- ralph/step-2  ...
```

Create branches manually before each increment. After RALPH completes and you've reviewed the result, merge to main. This gives you a clean rollback point per increment.

### Best Practice for ReAgt

```bash
# Before each increment:
git checkout -b ralph/step-N
# Update fix_plan.md and specs
ralph --monitor
# After completion:
# Review code, run tests manually
git checkout main
git merge ralph/step-N
```

---

## 8. Alternatives & Comparison

### RALPH vs. Raw Claude Code (No Loop)

Without RALPH, you'd manually invoke `claude` for each task, review, then invoke again. RALPH automates this to "set it and forget it" within an increment. For ReAgt, this saves hours of manual back-and-forth per step.

**Trade-off**: Raw Claude Code gives you review between each invocation. RALPH doesn't. For safety-critical code (which ReAgt isn't), raw Claude Code might be preferable.

### RALPH vs. Official Anthropic Ralph Wiggum Plugin

The Anthropic plugin uses a **stop hook** (intercepts Claude's exit attempts, keeps the session going). RALPH uses a **bash loop** (fresh invocation per iteration, state in files).

| Aspect | Anthropic Plugin | frankbria/RALPH |
|---|---|---|
| Context continuity | Same session (accumulates) | Session resume with fresh prompt |
| State tracking | In-context only | Files + git + circuit breaker |
| Exit detection | Hook-based | Dual-condition gate + 6 criteria |
| Stagnation protection | None | Circuit breaker |
| Rate limiting | None | Built-in |
| Monitoring | None | tmux dashboard |

**For ReAgt**: Use frankbria/RALPH. The safety mechanisms (circuit breaker, exit detection) matter for multi-hour autonomous runs.

### RALPH vs. Aider

Aider is 80-98% cheaper per feature and has deeper git integration. But it doesn't have RALPH's autonomous loop with exit detection. For ReAgt, where the goal is overnight autonomous execution per increment, RALPH is the better fit. If cost is a major concern, Aider is worth investigating for simpler increments.

### RALPH vs. Other Ralph Implementations

The ecosystem has 20+ Ralph-style tools. Notable alternatives:

| Tool | Differentiator |
|---|---|
| `choo-choo-ralph` | 5-phase workflow, compounding knowledge |
| `iannuttall/ralph` | Minimal, supports Codex/Claude/OpenCode |
| `ralph-orchestrator` | Rust, 7 AI backends, specialized personas |
| `smart-ralph` | Spec-driven with research/requirements/design phases |
| Vercel `ralph-loop-agent` | TypeScript, AI SDK integration |

**For ReAgt**: `frankbria/ralph-claude-code` remains the most feature-complete and battle-tested option with 6k+ stars, 484 tests at 100% pass rate, and active development.

### RALPH vs. Building Your Own Loop

Some power users write custom loop scripts instead of using RALPH. The value proposition of RALPH is the ~2,000 lines of battle-tested exit detection, circuit breaking, and response analysis you don't have to write yourself. For ReAgt, use RALPH rather than reinventing these wheels.

---

## 9. Common Mistakes When Using RALPH (and How to Avoid Them)

### Mistake 1: Vague fix_plan.md

**Problem**: Tasks like "implement the agent core" are too broad. Claude can't complete them in one iteration, leading to infinite loops.

**Fix**: Break every task into something achievable in 5-15 minutes of Claude work. "Create athlete_profile.py with a `load_profile(path) -> dict` function that reads JSON" -- that's one iteration.

### Mistake 2: No testable acceptance criteria

**Problem**: RALPH uses binary gates. "Make the agent smart" is not verifiable.

**Fix**: Every task needs a gate: a test that passes, a file that exists, a command that returns 0. Write these into your specs.

### Mistake 3: Not reviewing between increments

**Problem**: RALPH can't judge code quality. Claude might produce working but architecturally poor code. After 5 increments, you have a mess.

**Fix**: Always review after each RALPH run completes. Refactor before starting the next increment. This is your job, not RALPH's.

### Mistake 4: Too-broad ALLOWED_TOOLS

**Problem**: Granting `Bash(*)` lets Claude run arbitrary commands. In an autonomous loop, this is dangerous.

**Fix**: Be specific: `Bash(git *),Bash(python *),Bash(pytest *),Bash(pip *)`. Only grant what's needed.

### Mistake 5: Ignoring circuit breaker opens

**Problem**: When the circuit breaker opens, something is wrong. Setting `CB_AUTO_RESET=true` and restarting blindly wastes API credits.

**Fix**: When CB opens, read `logs/claude_output_*.log` for the last few iterations. Understand why progress stopped. Fix the underlying issue (missing dependency, unclear spec, broken test) before restarting.

### Mistake 6: Running the entire roadmap as one increment

**Problem**: Putting all 6 steps of ReAgt into a single fix_plan.md. Claude will lose context, skip steps, or produce shallow implementations.

**Fix**: One step = one RALPH run. Between runs, review and prepare the next increment. This is how power users achieve quality at scale.

### Mistake 7: Not using session continuity

**Problem**: Without `SESSION_CONTINUITY=true`, each iteration starts completely fresh. Claude re-explores the entire codebase every time.

**Fix**: Enable session continuity (default in RALPH). Within a 24h session, Claude remembers what it did and builds on it.

### Mistake 8: Expecting RALPH to handle environment setup

**Problem**: API keys, database connections, system dependencies -- these need human setup.

**Fix**: Handle all environment setup manually before the first `ralph` invocation. Document what's needed in AGENT.md. Make it a prerequisite, not a task.

### Mistake 9: No cost cap

**Problem**: A runaway loop (circuit breaker misconfigured) can consume hundreds of dollars.

**Fix**: Set `MAX_CALLS_PER_HOUR` conservatively (50 for experimentation, 100 for production runs). Monitor the tmux dashboard.

### Mistake 10: Treating RALPH output as production-ready

**Problem**: "Ralph built it, so it works" -- no. RALPH verified that Claude said it works and that git diffs exist. That's not the same as human code review.

**Fix**: Every RALPH output is a draft. Review it as you would review a junior developer's PR.

---

## 10. Summary: The ReAgt + RALPH Execution Plan

| Phase | Action | Duration | Who |
|---|---|---|---|
| Prep | Write PROMPT.md, AGENT.md, .ralphrc, .env | 30 min | Human |
| Step 0 | Write fix_plan.md for setup, run `ralph --monitor` | ~30 min | RALPH |
| Review | Check code, verify tests, merge to main | 15 min | Human |
| Step 1 | Update fix_plan.md for Dumb Coach, run `ralph` | ~45 min | RALPH |
| Review | Check Gemini integration, plan quality, merge | 20 min | Human |
| Step 2 | Update for Perceiving Coach, add FIT parser spec, run `ralph` | ~1-2 hrs | RALPH |
| Review | Test with real FIT files, review parsing quality, merge | 30 min | Human |
| Step 3 | Update for Adaptive Coach, add state machine spec, run `ralph` | ~2-3 hrs | RALPH |
| Review | Test adaptation logic, review autonomy categorization, merge | 30 min | Human |
| Step 4 | Update for Reflecting Coach, add episodic memory spec, run `ralph` | ~2-3 hrs | RALPH |
| Review | Test reflection quality, review episode storage, merge | 30 min | Human |
| Step 5 | Update for Proactive Coach, add trajectory spec, run `ralph` | ~2-3 hrs | RALPH |
| Review | End-to-end testing, final review | 1 hr | Human |

**Total estimated**: ~12-16 hours of RALPH execution + ~4 hours of human review = a working MVP of ReAgt.

The key discipline: **incremental delivery with human review gates between each step**. RALPH handles the grind. You handle the judgment.

---

*This document is based on analysis of frankbria/ralph-claude-code v0.11.4, community best practices as of February 2026, and the ReAgt project specification.*
