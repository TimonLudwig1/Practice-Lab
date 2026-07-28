# CLAUDE.md — Web Engineering Lab

This repository is a **learning workshop**. Your job as Claude Code is to
independently produce high-quality learning material for each of the 17 modules
in `MODULE_OVERVIEW.md`: a thorough theory script (`THEORY.md`) plus three
projects that build on each other (01-basic, 02-medium, 03-final).

The user works with expensive models. **Token efficiency comes first.**

---

## The three steering files

| File | Role |
|---|---|
| `PROGRESS.md` | Single source of truth for progress. Holds the `NEXT ACTION` pointer. |
| `MODULE_OVERVIEW.md` | Content guide: working instructions at the top, then one section per module. |
| `CLAUDE.md` (this file) | Entry point and quick reference. Deliberately holds no detail that lives elsewhere. |

---

## Session flow

1. Read `PROGRESS.md` — header and legend only, never the whole session log.
2. Follow the `NEXT ACTION` pointer.
3. From `MODULE_OVERVIEW.md`, read **only** the section for that module. On the
   first task of a session, also read the working instructions at the top.
4. Work on exactly **one** unit (one theory script or one project).
5. Run the code and confirm it works.
6. Update `PROGRESS.md`: set the status, move the pointer to the next action, add
   one line to the session log.
7. Commit. Then take the next unit or end the session.

Move the pointer **before** starting a new unit, so an interrupted run never
leaves stale state behind.

---

## Core rules

1. **Never reopen finished work.** Units marked `[x]` are closed and do not get
   polished unless the user explicitly asks.
2. **One task per cycle.** No working ahead into other modules.
3. **Raw form before framework.** An abstraction is introduced only after the
   underlying mechanism has been made visible. Details at the top of the module
   overview.
4. **English only.** Documentation, theory scripts, project briefs, code,
   docstrings, comments, UI text and commit messages are all English. No emojis.
5. **Do not ship solutions.** Projects consist of a task, scaffolding, TODO
   markers and criteria. Reference implementations go under `solution/` only, and
   the project readme mentions them without explaining them.
6. **"Done when" instead of time estimates.** Criteria are verifiable, never
   measured in hours.
7. **Code runs.** Every project gets started and checked before it counts as
   finished. Datasets come from `shared/data/` with a fixed seed and the reasoning
   written as a comment.
8. **No secrets.** `.env.example` instead of `.env`, no keys in examples, no
   personal data in logs.
9. **When unsure about depth:** thorough and beginner-friendly beats terse. Make
   real design decisions yourself (directory layout, library choice, notebook or
   script) and justify each in one sentence in the project readme.

---

## Acting as a tutor

If the user is **working through** a project rather than having it built: let them
work first, then help. Hints before solutions; look at and comment on their
attempt before showing anything. For web errors, do not hand over the fix — walk
them through the diagnosis: network tab, console, server log, status code.

---

## First run

If they do not exist yet, create `modules/` and `shared/`, then go straight to the
next action named in `PROGRESS.md`. No further initialization needed; the steering
files are complete.
