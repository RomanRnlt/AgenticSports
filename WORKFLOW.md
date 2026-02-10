> **OUTDATED** -- This document is no longer actively maintained. The workflow described here has been superseded by the practices documented in CLAUDE.md and the project's git history. Kept for historical reference only.

# ReAgt: Development Workflow & Execution Strategy (v2)

## How to Build This Project with Claude Code

*February 2026 | Companion to REAGT.md*

---

## 1. Critique of the Previous Workflow

The previous version of this document (v1) was written by synthesizing blog posts, framework guides, and Anthropic's documentation at face value. After deeper research into how experienced developers -- including Claude Code's creator -- actually work, several of v1's recommendations turn out to be cargo-cult patterns from early 2024 that add overhead without corresponding value.

### What Was Wrong

**PROGRESS.md is unnecessary overhead.** The previous workflow proposed a living PROGRESS.md file updated after every session, with checklists, session logs, and decision records. This pattern is not used by any experienced Claude Code developer documented in publicly available sources. Not by Boris Cherny (Claude Code's creator). Not by Anthropic's internal teams. Not by incident.io, Builder.io, or other teams that have published their workflows. Git history does this job better. Commits are durable, semantic, reviewable, and do not consume context tokens at session start. PROGRESS.md is a manual workaround for a problem that git already solves.

**The session ritual was too heavy.** V1 prescribed an elaborate start/during/end ritual: read PROGRESS.md, follow Explore-Plan-Code-Commit for every sub-task, update PROGRESS.md before closing. This treats sessions as precious, expensive resources that must be carefully managed. In practice, effective Claude Code users treat sessions as cheap and disposable. Boris Cherny runs 10-15 concurrent sessions and abandons 10-20% of them. The previous workflow's ceremony would make that impossible.

**Explore-Plan-Code-Commit was presented as mandatory.** V1 said "skip [the plan] and you'll spend more time correcting drift than you saved." Anthropic's own documentation says the opposite: "If you could describe the diff in one sentence, skip the plan." Plan Mode is useful for complex, uncertain tasks. It is overhead for clear, scoped tasks. The skill is knowing when to plan, not always planning.

**Sub-agent guidance was too restrictive.** V1 said "you'll use sub-agents rarely -- perhaps 10-15% of the time." The research shows sub-agents for investigation and codebase exploration are "one of the most powerful tools available" (Anthropic docs). They should be used routinely for exploration, not rarely.

**File-heavy state tracking is a 2024 pattern.** V1 proposed CLAUDE.md + WORKFLOW.md + PROGRESS.md, with the latter two imported via `@` syntax into CLAUDE.md. This consumes context tokens at every session start. Claude Code's auto memory system (writing to `~/.claude/projects/<project>/memory/MEMORY.md`) now handles cross-session learning automatically. Manual tracking files duplicate work that the tool already does.

**The per-step session prescriptions were too rigid.** V1 specified exact sessions for each roadmap step ("Session 1: Explore + Plan. Session 2: Implement."). Real development doesn't follow a script. A step might take one session or five, depending on what you discover. Prescribing session boundaries in advance is like writing a detailed schedule for exploratory research -- it creates the illusion of control at the cost of flexibility.

### What Was Correct

V1 got several things right:

- **Context is the scarcest resource.** This is the correct foundational constraint.
- **CLAUDE.md should be lean.** The recommendation to keep it under ~40 lines of high-signal content was good.
- **Tests as the verification loop.** "Tests, not vibes" is correct and is explicitly called "the single highest-leverage thing you can do" by Anthropic.
- **Don't use a framework from the start.** Building the core loop manually for a learning project remains the right call.
- **The troubleshooting guidance.** "/clear after 2 failed corrections" and "commit often" are both validated by research.
- **Anti-patterns list.** Kitchen Sink Sessions, Agent Swarms, and Premature Abstraction are real and worth naming.

---

## 2. Research-Backed Summary of Modern Claude Code Development

### How Experienced Developers Actually Work

Based on published workflows from Boris Cherny (Claude Code creator), Anthropic's internal teams, incident.io, Shrivu Shankar, and Steve Sewell (Builder.io):

**Sessions are cheap, not precious.** The dominant pattern is running multiple parallel sessions (5-15), treating each as a disposable experiment. When one goes wrong, abandon it. When one works, commit and move on. This requires separate git checkouts or worktrees so sessions don't conflict.

**Git is the state system.** Commits are the progress tracker. Branch diffs are the progress file. `git log --oneline` is the session log. Conventional commits (`feat:`, `fix:`, `refactor:`) make this machine-readable. There are no PROGRESS.md files in any documented expert workflow.

**CLAUDE.md is a correction log, not a project manual.** Anthropic's own CLAUDE.md is ~2.5k tokens. The team adds to it whenever Claude makes a mistake in the project. It is not a comprehensive description of the project -- it is a targeted list of things Claude gets wrong without being told.

**Plan when uncertain, skip when clear.** Boris Cherny uses Plan Mode for every PR -- but he also auto-accepts edits once the plan is good. For a solo developer on a learning project, use Plan Mode for architectural decisions and unfamiliar territory. For clear tasks where you know what the diff should look like, just ask Claude to do it.

**Verification is non-negotiable.** Every practitioner emphasizes this. Tests, linters, type checkers -- anything that lets Claude check its own work. Anthropic says this single practice improves results by 2-3x.

### The Ralph Pattern: Autonomous Loops

Geoffrey Huntley's "Ralph" technique -- running Claude Code in a loop via `claude -p` -- has become a recognized pattern with multiple implementations. At its simplest:

```
while true; do claude -p "Read TASKS.md, implement the top item, test, commit, update TASKS.md" ; done
```

More sophisticated implementations (frankbria/ralph-claude-code, Continuous Claude, Autoloop) add circuit breakers, session continuity, exit detection, and cost tracking.

**Assessment for this project:** Ralph-style loops solve a real problem -- sustained autonomous development without human attention. However, for a *learning project* where understanding matters more than throughput, they abstract away exactly the interactions that produce learning. The value of working with Claude interactively is that you see decisions being made and can question them. An autonomous loop optimizes for output, not understanding.

**When Ralph makes sense for this project:** Steps 0-1 (project setup, basic coach) have well-defined scope and simple success criteria. Running these semi-autonomously could save time without sacrificing learning. Steps 3-4 (adaptive coach, reflecting coach) require judgment and architectural decisions that benefit from interactive exploration.

The right pattern is not "always interactive" or "always autonomous" -- it is autonomous for mechanical tasks, interactive for decisions that require understanding.

### Git as the Primary Coordination System

Git replaces external state management for AI-assisted development:

| Traditional Tracking | Git-Based Equivalent |
|---|---|
| PROGRESS.md with checkboxes | Commit history + branch diff |
| Session log table | `git log --oneline --since="2 days ago"` |
| "Decisions Made" section | Commit message bodies with rationale |
| Handoff documents | Branch state + CLAUDE.md + `--continue` |
| Known issues list | Open issues in git platform, or `# TODO` comments |

**How Claude Code uses git:** It does not proactively read git history at session start. But when generating commits, it reads recent history and diffs to match your style. The session picker organizes by repo and branch. Auto memory is scoped per git repository. When you ask Claude about project history, it runs `git log`, `git show`, `git blame` on demand.

**The practical implication:** You don't need to tell Claude what happened. You need to make what happened discoverable via git. Conventional commits, clear branch names, and descriptive PR bodies are the mechanism.

### Context Management: What Actually Works

- **`/clear` aggressively.** Between unrelated tasks. After 2 failed corrections. When context feels polluted.
- **Sub-agents for exploration.** When Claude needs to read many files or run verbose commands, delegate to a sub-agent so the output doesn't fill your main context.
- **Auto-compaction triggers at ~75% capacity.** It preserves code patterns, file states, key decisions, and structured data (plans, to-do lists). It may lose detailed early instructions and nuanced constraints mentioned once. Put critical instructions in CLAUDE.md, not conversation.
- **Plans survive compaction better in files than in conversation.** If you create a plan, write it to a markdown file. If it's only in conversation, compaction may lose it.
- **Opus 4.6 has a 1M token context window.** This reduces urgency but doesn't eliminate context rot. Sessions still degrade over time -- it just takes longer.

---

## 3. The Improved Workflow

### Principles

1. **Git is the source of truth for project state.** No separate tracking files.
2. **CLAUDE.md is a correction log.** Add things when Claude gets them wrong. Prune when it stops making those mistakes.
3. **Sessions are cheap.** Start new ones freely. Abandon ones that go wrong. Don't over-invest in session rituals.
4. **Plan when uncertain, act when clear.** Match the level of ceremony to the level of uncertainty.
5. **Tests verify everything.** Claude can check its own work if you give it tests. This is the single highest-leverage practice.
6. **The human's role is judgment, not bookkeeping.** Review architectural decisions, verify domain correctness, approve plans. Don't maintain tracking files.

### Project Structure

```
reagt/
+-- CLAUDE.md                  # Lean correction log + build commands (<100 lines)
+-- REAGT.md                   # Architecture & roadmap (source of truth, read on demand)
+-- src/
|   +-- agent/
|   +-- tools/
|   +-- memory/
|   +-- interface/
+-- data/
|   +-- fit_files/
|   +-- athlete/
|   +-- plans/
|   +-- episodes/
+-- tests/
+-- pyproject.toml
```

What's gone vs v1: WORKFLOW.md is reference material, not loaded by Claude. PROGRESS.md is eliminated entirely. `.claude/rules/` is premature -- add it when you have enough rules to organize. The structure is minimal: code, tests, data, and one instruction file.

### CLAUDE.md

```markdown
# ReAgt - Adaptive Training Agent

## Project
Python 3.11+, Gemini API backend, CLI interface.
Learning project -- prefer clarity over cleverness.

## Commands
- Install: `uv sync`
- Test: `pytest tests/`
- Lint: `ruff check src/`
- Type check: `mypy src/`

## Architecture
See REAGT.md for full architecture.
- State machine agent loop (IDLE -> PERCEIVE -> REASON -> PLAN -> PROPOSE -> EXECUTE -> REFLECT)
- Memory: JSON files in data/
- Tools: Pure functions with typed I/O in src/tools/
- LLM: Gemini API via google-genai SDK

## Conventions
- Type hints on all public functions
- Tests first -- write the test, then the implementation
- No abstractions until 3+ concrete use cases
- Conventional commits: feat/fix/refactor/test/docs

## Mistakes to Avoid
(Add entries here when Claude gets something wrong in this project)
```

Under 30 lines. The "Mistakes to Avoid" section starts empty and grows organically. This is how Anthropic's own team maintains their CLAUDE.md.

### How Sessions Work

**Starting a session:**

```
cd reagt/
claude
```

That's it. Claude loads CLAUDE.md automatically. If you need context on what you did last, ask Claude:

```
"Run git log --oneline -20 and git diff main to show me where we are."
```

Or use `claude --continue` to resume the most recent session. Or `claude --resume` to pick from named sessions.

**During a session -- for clear, scoped tasks:**

```
You: "Add a parse_fit_file function in src/tools/fit_parser.py that extracts
     duration, distance, avg HR, avg pace from a FIT file using fitdecode.
     Write a test first using the sample file in data/fit_files/."

Claude: [writes test, implements function, runs tests, reports results]

You: [review the code, verify it makes sense]

You: "Commit this."
```

No Plan Mode. No Explore phase. The task is clear. Let Claude work.

**During a session -- for uncertain or complex tasks:**

```
You: "I need to implement the state machine from REAGT.md section 2.2.
     Read that section and the existing code in src/agent/.
     Before writing anything, propose how you'd implement this."

Claude: [reads files, proposes approach]

You: [review, ask questions, adjust]

You: "Good. Implement it."
```

This is planning without Plan Mode's formal mechanism. If you want the guarantee that Claude won't accidentally start editing while you discuss, use Plan Mode (Shift+Tab). Otherwise, natural language works fine.

**Ending a session:**

```
You: "Commit everything with a descriptive message."
```

That's it. No PROGRESS.md update. No session log entry. The commit IS the progress marker. If you're mid-work and need to stop, commit what works as a WIP:

```
You: "Commit what we have as 'wip: partial state machine implementation'"
```

Next session, `git log` and `git diff` tell you exactly where you are.

**When to start a new session vs continue:**

| Situation | Action |
|---|---|
| Continuing the same task, same day | `claude --continue` |
| Starting a new task | Fresh session (`claude`) or `/clear` |
| Context feels polluted (Claude confused, repeating itself) | `/clear` or new session |
| After 2 failed correction attempts | `/clear` with a better prompt |
| Switching to a different roadmap step | New session |

### How Claude Decides "What to Do Next"

Claude doesn't decide -- you do. At the start of each session, you tell Claude what to work on based on:

1. **The roadmap in REAGT.md** -- which step are you on?
2. **Git state** -- what's committed, what's in progress?
3. **Your judgment** -- what matters most right now?

This is appropriate for a learning project. If you wanted autonomous task selection, you'd use the Ralph pattern -- but that optimizes for output, not understanding.

For the mechanical parts of the roadmap (project setup, boilerplate), you *can* let Claude drive:

```
You: "Read REAGT.md Step 0. Set up the project exactly as described.
     Create the directory structure, pyproject.toml, and a test that verifies
     Gemini API connectivity. Commit when done."
```

For the learning-heavy parts (state machine, reflection, episodic memory), drive interactively:

```
You: "Read REAGT.md Step 3. I want to understand how the state machine works.
     Walk me through how you'd implement PERCEIVE -> REASON -> PLAN transitions.
     Don't write code yet."
```

### When to Use Sub-Agents

**Use sub-agents for:**

- **Codebase exploration.** "Use a sub-agent to find all places where the athlete profile is read or written." This keeps verbose search results out of your main context.
- **Code review.** "Use a sub-agent to review src/agent/states.py for edge cases and potential bugs." Fresh context prevents confirmation bias.
- **Running noisy commands.** Test suites, linters, or any command that produces pages of output. The sub-agent returns a summary.

**Don't use sub-agents for:**

- Quick file reads or small edits (the 20k token startup overhead isn't worth it).
- Implementation work (sub-agents lack your conversation context).
- Anything that needs back-and-forth iteration.

### Git Strategy

**Branching model:**

```
main              Stable, tested code. Each roadmap step merges here when done.
  +-- step-1      Working branch for Step 1 (The Dumb Coach)
  +-- step-2      Working branch for Step 2 (The Perceiving Coach)
  +-- experiment/* Branches for risky experiments (discard if they fail)
```

**Commit conventions:**

```
feat(tools): add FIT file parser with duration/distance/HR extraction
test(tools): add FIT parser tests using real garmin data
refactor(agent): extract assessment logic from coach into separate module
fix(metrics): correct TRIMP calculation for recovery HR
docs: update REAGT.md with implementation learnings from Step 2
wip: partial state machine -- PERCEIVE and REASON states working
```

The commit history becomes your project narrative. Each commit is a decision point with a diff attached. No separate documentation needed.

**When to commit:**

- After every passing test suite (this is your save point)
- Before any risky refactor (`git stash` or commit first)
- At the end of every session, even if incomplete (use `wip:` prefix)
- Use `git stash` for quick experiments you might discard

---

## 4. Executing the MVP Roadmap

The per-step checklists from v1 were reasonable but too rigid in their session prescriptions. Here's a lighter-weight version that preserves the "done" criteria without prescribing session boundaries.

### Step 0: Project Setup

**Goal:** A Python project that can call Gemini and has tests.

**Claude's job:** Create pyproject.toml, directory structure, a minimal LLM client, and a test that proves API connectivity. Initialize git. Create CLAUDE.md.

**Your job:** Verify `pytest` passes. Verify the directory structure matches REAGT.md. Verify CLAUDE.md is accurate.

**Done when:** `pytest tests/` passes. Git repo initialized. Gemini returns a response.

**This step is mechanical.** Let Claude do it in one shot. A single prompt covers it:

```
"Read REAGT.md Step 0 and Appendix. Set up the project as described.
Use uv for dependency management. Write a test that calls Gemini and verifies
a non-empty response. Initialize git. Create CLAUDE.md per the template
in REAGT.md. Commit everything."
```

### Step 1: The Dumb Coach

**Goal:** CLI where you describe your goals, agent creates athlete profile, generates 1-week training plan.

**Claude's job:** Implement profile read/write, system prompt for coaching, Gemini function calling for plan generation, minimal CLI.

**Your job:** Verify the training plan makes athletic sense. Verify function calling is reliable (run it 3 times). Verify JSON schemas are clean.

**Done when:** CLI input -> profile created -> plan generated -> plan saved to disk. Plan makes sense as coaching.

**Key validation:** This is where you discover if Gemini's function calling and structured output work reliably. If they're flaky, adjust the approach now.

### Step 2: The Perceiving Coach

**Goal:** Agent parses FIT files, extracts metrics, stores structured training log, summarizes training week.

**Claude's job:** Implement FIT parser, metric calculations (TRIMP, zones), training log storage.

**Your job:** Compare parsed data against Garmin/Strava for the same file. Verify TRIMP values are reasonable. Verify the weekly summary is factually accurate.

**Done when:** Drop a FIT file -> parse -> metrics calculated -> stored -> accurate weekly summary displayed.

**Key validation:** Does the parsed data match reality? This is domain verification that only you can do.

### Step 3: The Adaptive Coach

**Goal:** Agent compares planned vs actual training, proposes adjustments, categorizes decisions by impact.

**Claude's job:** Implement the state machine (PERCEIVE -> REASON -> PLAN -> PROPOSE), assessment logic, autonomy classification.

**Your job:** Verify state transitions are correct. Verify assessments identify real issues. Verify autonomy classification makes sense. Verify adjusted plans are good coaching.

**Done when:** Upload a FIT file showing you ran too fast on an easy day. Agent detects this, proposes adjustment, explains reasoning, waits for approval.

**This is the hardest step.** Work interactively. Discuss the state machine design before implementing. Ask Claude to walk you through transitions. This is where understanding matters most.

### Step 4: The Reflecting Coach

**Goal:** Agent generates structured reflections after training blocks, uses past reflections when planning.

**Claude's job:** Implement reflection generation, episode storage/retrieval, planning with episode context.

**Your job:** Verify reflections capture genuine insights (not just restating data). Verify plans with episodes differ meaningfully from plans without.

**Done when:** After 2+ training blocks, reflections capture real patterns and subsequent plans reference them.

**Key validation:** This is the most research-relevant step. The quality of reflections determines whether the agent actually learns. Read them carefully.

### Step 5: The Proactive Coach

**Goal:** Agent projects trajectory toward long-term goal, communicates proactively with calibrated confidence.

**Claude's job:** Implement trajectory projection, confidence scoring, proactive CLI assessments.

**Your job:** Verify projections are mathematically sound. Verify confidence correlates with data quality. Verify proactive messages are useful.

**Done when:** On startup, agent proactively assesses progress toward the half marathon goal with explicit confidence levels.

---

## 5. Why This Workflow Is Better Than My Previous Proposal

### It eliminates tracking overhead that nobody actually uses.

No PROGRESS.md. No session logs. No handoff documents. Not because these sound bad in theory, but because no experienced Claude Code developer uses them. Git history and CLAUDE.md are sufficient. Every extra file loaded at session start consumes context tokens and increases the chance that Claude ignores your actual instructions in favor of stale tracking information.

### It treats sessions as cheap, not precious.

V1's elaborate start/end rituals assumed each session is expensive and must be carefully managed. The research shows the opposite: effective users treat sessions as disposable experiments. Start many, abandon the ones that don't work, commit from the ones that do. This requires less ceremony, not more.

### It matches planning effort to task uncertainty.

V1 mandated Explore-Plan-Code-Commit for everything. The improved workflow uses planning for complex or uncertain tasks and skips it for clear, scoped ones. This is what Anthropic recommends and what experienced users do.

### It uses git as the state system instead of duplicating it.

Commits are progress markers. Branch diffs show current state. Conventional commit messages document decisions. `git log` is the session log. This is not a workaround -- it is the native mechanism. PROGRESS.md was a parallel system that had to be kept in sync with git, adding maintenance burden with no unique information.

### It acknowledges that Claude Code has improved since early 2024.

Auto memory, session memory, auto-compaction at 75%, 1M token context on Opus 4.6 -- the tool has evolved. Patterns that were necessary workarounds in 2024 (elaborate context files, manual state tracking, careful context budgeting) are now partially or fully automated. The workflow should match the tool's current capabilities.

### It is honest about the autonomous loop option.

V1 didn't mention the Ralph pattern at all. The improved workflow acknowledges it exists, evaluates its trade-offs, and recommends it for mechanical tasks while keeping interactive work for learning-heavy tasks. This is a more honest assessment of the options than pretending interactive-only is the only approach.

### It is shorter.

V1 was 569 lines. This document is shorter because it eliminated redundant tracking systems, prescriptive session scripts, and ceremony that added process without adding value. A shorter workflow is easier to follow, which means it's more likely to actually be followed.

---

*This document is a reference for how to work on the ReAgt project. It is not loaded by Claude at session start -- CLAUDE.md is the only file Claude reads automatically. This document is for you, the human, to consult when you need to think about process.*
