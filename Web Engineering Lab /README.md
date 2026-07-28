# Web Engineering Lab

A hands-on self-study lab covering the full web stack: frontend with TypeScript
and React, backend with Python and FastAPI, plus the parts that usually get
skipped — HTTP itself, persistence, security, testing, deployment. Part of the
[`Practice-Lab`](../) collection.

## What makes this lab different

Most web tutorials start with a framework and never explain what it replaced.
This one goes the other way around. Every module builds the raw version first:
you inspect requests with `curl` before touching `fetch`, render a list with
`document.createElement` before React shows up, and write a response handler on
`http.server` before FastAPI does it for you. Only then does the abstraction
arrive, together with the question of what exactly it took off your hands.

The three-level structure applies to every theory script: intuition, then a
hands-on raw build of the mechanism, then the formal definition. Never the other
way around.

## Stack

| Layer | Tools |
|---|---|
| Frontend | HTML, CSS, TypeScript, Vite, React 18, React Router, TanStack Query, Tailwind, Recharts |
| Backend | Python 3.11+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, Alembic |
| Data | SQLite locally, PostgreSQL in Docker, Redis for caching |
| Testing | pytest, httpx, Vitest, React Testing Library, MSW, Playwright |
| Operations | Docker, Docker Compose, GitHub Actions |

Why Python on the backend: for general-purpose web backends, Node and TypeScript
lead the market. For data-adjacent backends — model serving, analytics APIs,
internal data tooling — Python with FastAPI or Django is the norm, and that is
what this lab targets. Node still shows up, because the entire frontend toolchain
runs on it.

## Structure

17 modules in six blocks:

| Block | Modules | Focus |
|---|---|---|
| A | 01–04 | Web fundamentals: HTTP, HTML, CSS, JavaScript |
| B | 05–08 | Frontend engineering: tooling, TypeScript, React, UI state |
| C | 09–12 | Python backend: FastAPI, REST design, persistence, security |
| D | 13–15 | Integration and quality: full stack, testing, performance |
| E | 16 | Operations: containers, deployment, CI/CD, observability |
| F | 17 | Capstone: a complete data application |

Each module contains a theory script (`THEORY.md`) and three projects that build
on each other: `01-basic` implements the mechanism by hand, `02-medium` takes the
framework path and explores variants, `03-final` solves a realistic problem with
a link to data science work.

```
web-engineering-lab/
├── CLAUDE.md              Entry point for Claude Code sessions
├── PROGRESS.md            Single source of truth for build progress
├── MODULE_OVERVIEW.md     Content guide for all modules
├── shared/                Reusable utilities and data generators
└── modules/
    └── 01_web_fundamentals/
        ├── THEORY.md
        ├── README.md
        ├── 01-basic/
        ├── 02-medium/
        └── 03-final/
```

## Getting started

```bash
git clone <repository-url>
cd web-engineering-lab
```

Read `PROGRESS.md` first: the `NEXT ACTION` pointer at the top says what comes
next. Then work through the current module's `THEORY.md` and start with
`01-basic`.

Requirements: Python 3.11 or newer, Node.js 20 or newer, Git, Docker from module
16 onwards, and a browser with working developer tools.

## Notes

Everything here is written in English: documentation, code, comments, commit
messages. Progress in `PROGRESS.md` tracks how much content has been generated,
not how much has been learned. No solutions are shipped up front — where a
reference implementation exists, it sits in `solution/` and is meant to be opened
after your own attempt.
