# llm/ — LLM Context Directory

This directory contains structured context, state, and history files to help an LLM agent
quickly onboard and work effectively on this project across sessions.

## Directory Structure

```
llm/
├── AGENT_START.md          — Read this first: project overview, user profile, anti-patterns
├── AGENT_END_ROUTINE.md    — Checklist to run at the end of every session
├── .llmignore              — Files excluded from LLM context loading
├── context/
│   ├── start_prompt.md     — Default opening prompt for new sessions
│   ├── architecture.md     — Planned module structure and interfaces
│   ├── code_snippets.md    — Key code patterns and examples
│   └── dependencies.md     — External dependencies and versions
├── state/
│   ├── current.md          — What is done / in progress / broken right now
│   ├── plan.md             — Step-by-step implementation plan with status
│   ├── issues.md           — Known bugs and blockers
│   └── session_handoff.md  — What to do next at the start of the next session
├── history/
│   ├── changelog_recent.md — Recent changes (last ~3 sessions)
│   ├── changelog_archive.md— Older history (excluded from default context)
│   └── decisions.md        — Key design decisions and their rationale
└── tools/
    └── get_context.py      — CLI helper: routes task type to relevant context files
```

## Quick Start for a New Session

1. Read `AGENT_START.md`
2. Read `state/current.md` and `state/session_handoff.md`
3. Read `state/plan.md` to know what step comes next
4. Run `python llm/tools/get_context.py <task-type>` for targeted context
