# Module Overview — Web Engineering Lab (Frontend & Backend)

This file is the **content guide** for the Claude Code session that builds the
`web-engineering-lab`. It defines every module, its theory content, learning
objectives, and the three practice projects per module.

**Token discipline:** per work cycle, read only the section of the module
currently being built. Never re-read the whole file, never re-read sections of
finished modules.

---

## Working instructions for the Claude Code session

### Language and style

Everything in this lab is written in **English**: documentation, theory scripts,
project briefs, code, docstrings, comments, UI text, and commit messages. No
German anywhere, no mixed-language files.

Professional tone, **no emojis**. Prose over bullet lists where a sentence does
the job. Explain like a good lecturer, not like a reference manual: the reader is
a data science student who can program in Python but has never built a web
application.

### Didactic principle: raw form before framework

Web development is usually learned backwards — React first, and the question of
what an HTTP request actually is comes years later, if ever. This lab inverts
that. Every theory script follows a three-level structure:

1. **Intuition.** What problem existed before this technique? What concretely
   happens between browser and server, which file lives where, who renders what?
2. **Raw build.** Make the mechanism visible once without abstraction: inspect
   requests with `curl` and the browser DevTools before using `fetch`; render a
   list with `document.createElement` before React appears; write a response
   handler on `http.server` before FastAPI does it for you; write the SQL by hand
   before the ORM generates it.
3. **Formalization.** Only now: specification, framework API, patterns and
   conventions (REST, the component model, ASGI, layered architecture), plus an
   explicit comparison — raw build versus framework, what exactly was taken off
   my hands?

When an abstraction is introduced, the script must show **what it replaces**.
Framework magic without a preceding raw build is a defect, not a shortcut.

### Stack (binding)

Deliberately close to what data-adjacent teams actually run.

**Frontend:** HTML, CSS and vanilla JavaScript in Block A; from module 05 onward
TypeScript, Vite, React 18. Styling handwritten first (CSS, Flexbox, Grid), then
Tailwind from module 08. Charts with Recharts, client routing with React Router,
server state with TanStack Query.

**Backend:** Python 3.11+. Module 09 starts with `http.server` from the standard
library, then FastAPI with Uvicorn, Pydantic v2, SQLAlchemy 2.0 and Alembic
throughout. SQLite locally, PostgreSQL in a container from module 16.

**Testing:** pytest and httpx on the backend, Vitest with React Testing Library
on the frontend, Playwright for end-to-end.

**Operations:** Docker, Docker Compose, GitHub Actions, structured logging.

**Market context, spelled out in module 09:** general-purpose web backends are
dominated by Node/TypeScript, with Java and Go behind it. Python is present but
not the default there. In **data-adjacent** backends — model serving, analytics
APIs, internal data tooling, anything sitting on pandas, scikit-learn or a
warehouse connection — Python with FastAPI or Django is the normal choice. That
is what this lab targets, hence Python. Node still appears, because the entire
frontend toolchain runs on it; module 05 puts that in perspective.

### Per-module structure

Each module lives under `modules/NN_short_name/` and contains:

```
THEORY.md           Lecture script: intuition -> raw build -> formalization
README.md           Module readme: objectives, project order, prerequisites
01-basic/           Build the mechanism by hand, small scope
02-medium/          Framework path, application, variants
03-final/           Realistic problem with a link to data science work
```

Every project has its own `README.md` stating the task, requirements,
"Done when…" criteria, and hints — **no solution**. Provide scaffolding and TODO
markers. Where a reference implementation is genuinely useful, put it under
`solution/` and mention it in the project readme without explaining it.

**Role split:** the learner writes the code. Claude Code builds the task, the
scaffolding, the test cases and the criteria. Do not ship finished solutions
unless explicitly asked.

### Format decision

Web code belongs in files, not notebooks. The default deliverable is a runnable
project directory with `package.json` or `requirements.txt`. Notebooks only where
actual data analysis happens (modules 08 and 17). Conceptual comparisons that
never execute may be Markdown walkthroughs.

### Data

Synthetic datasets are generated locally under `shared/data/generate_*.py`, with
a fixed seed and the reasoning behind the distributions written as a comment in
the script. Prefer real open datasets where they are reachable without
registration. One domain runs through all modules: a **sensor and usage data
platform** (devices, readings, users, alerts), because it covers plain CRUD as
well as time series, aggregation, and model serving.

### Security and quality rules

No project prints passwords, no secrets in code, no API keys in examples.
`.env.example` instead of `.env`. From module 12 onward every input is untrusted,
and that applies retroactively to earlier projects that get extended.

---

## Block overview

| Block | Modules | Content |
|---|---|---|
| A | 01–04 | Web fundamentals: HTTP, HTML, CSS, JavaScript |
| B | 05–08 | Frontend engineering: tooling, TypeScript, React, UI state |
| C | 09–12 | Python backend: FastAPI, REST design, persistence, security |
| D | 13–15 | Integration and quality: full stack, testing, performance |
| E | 16 | Operations: containers, deployment, CI/CD, observability |
| F | 17 | Capstone: a complete data application |

---

# Block A — Web Fundamentals

## Module 01 — How the Web Works

### Theory
What happens between the address bar and a rendered pixel: DNS resolution, TCP
connection, TLS, HTTP request and response. Anatomy of a message (method, path,
headers, body), the status code classes, and why choosing 301 over 302 has
consequences you cannot undo. Idempotency and safety of methods. State in a
stateless protocol: cookies, sessions, caching headers. The difference between
static file serving, server-side rendering, and a single page application.
Browser DevTools as a measuring instrument: network tab, waterfall, payload
sizes, response times. HTTP/1.1 versus HTTP/2 in one paragraph, no deep dive.

### Objectives
Done when the learner can decompose a request into its parts and reproduce it
with `curl`; can name and justify the right status code for a given scenario; and
can explain from the network tab which request triggered which other request.

### Projects
- **01-basic — Request dissection.** Reproduce ten given scenarios (form
  submission, image load, login, 404, redirect) with `curl -v`, document the raw
  messages, explain the headers. Deliverable is a Markdown walkthrough containing
  the real output.
- **02-medium — Static server by hand.** A server built on Python's
  `http.server` that serves files, sets correct content types, handles 404, and
  issues a cache header. Then measure: what does the browser do with and without
  `Cache-Control`?
- **03-final — Latency report for a real API.** Call an open API (Open-Meteo, for
  example) 100 times, log response times and status codes, compute the
  distribution and percentiles, interpret the result. First bridge to data
  analysis: median versus mean for latency.

---

## Module 02 — HTML, Document Structure, Forms

### Theory
The document as a tree. Semantic elements and the concrete cost of div soup:
screen readers, SEO, maintainability. Block versus inline, attributes, references
between elements (`for`/`id`, `aria-*`). Forms in full: `method`, `action`,
`enctype`, the available field types, how browser validation works, and why it
never replaces server-side validation. Images, responsive sources, alternative
text. Accessibility as a default rather than a retrofit: keyboard operation,
focus order, contrast.

### Objectives
Done when a page is readable with CSS disabled and fully operable by keyboard;
when the learner can state which data a form sends in which format; and when
element choice is justified semantically rather than visually.

### Projects
- **01-basic — Semantic reconstruction.** Rebuild a deliberately badly marked-up
  page (nothing but `div` and `span`) with proper semantics, justify each choice
  in a comment, and verify the result in the accessibility tree.
- **02-medium — Form flow.** A multi-step registration form for the sensor
  platform using every relevant field type, fieldsets, error messages, and full
  keyboard operation. Make the submitted request visible using the raw server
  from module 01.
- **03-final — Report as a document.** A multi-page, purely static analysis page
  with tables, footnotes, figure captions and anchors, checked with an
  accessibility linter. No CSS layout, no JavaScript.

---

## Module 03 — CSS and Layout

### Theory
Cascade, specificity, inheritance — the three mechanisms behind most CSS
confusion. Box model and `box-sizing`. Normal flow, before anything gets
positioned. Flexbox for one axis, Grid for two, with a clear rule for choosing.
Units: `px`, `rem`, `%`, `vh`, `ch`, and why `rem` wins for typography.
Responsive design via media and container queries, mobile first. Custom
properties as design tokens. Positioning, stacking context, `z-index`.
Transitions briefly, animation only as an outlook.

### Objectives
Done when a layout works across three breakpoints without a framework; when the
learner can look at an unexpected style in DevTools and explain which rule wins
and why; and when Flexbox and Grid are chosen deliberately rather than by habit.

### Projects
- **01-basic — Layout catalogue.** Eight classic layout tasks (centering, holy
  grail, card grid, sticky header, scrolling sidebar), each solved twice — once
  with Flexbox, once with Grid — plus a note on which fits better.
- **02-medium — Dashboard shell, no logic.** A static dashboard layout with KPI
  tiles, chart placeholders and a table, responsive, driven by design tokens in
  custom properties, with a dark mode via `prefers-color-scheme`.
- **03-final — Refactor an existing page.** Give the module 02 form flow a
  complete stylesheet, then audit that stylesheet for duplication, move values
  into tokens, and document file size before and after.

---

## Module 04 — JavaScript in the Browser

### Theory
Language core as far as it is needed: types, `let`/`const`, scope, closures,
`this`, destructuring, spread, modules. Arrays functionally (`map`, `filter`,
`reduce`) as preparation for React. The DOM as an API: select, create, insert,
attributes versus properties. Events, bubbling, delegation, `preventDefault`. The
event loop, microtasks versus macrotasks — this is where the `setTimeout(…, 0)`
example belongs. Promises and `async`/`await`, `fetch`, JSON, error handling (why
a 404 is not a rejection), `AbortController`. Same-origin policy and CORS as a
preview of module 12.

### Objectives
Done when the learner correctly predicts the console output order of a mixed
sync/async example before running it; renders a list from API data without a
framework; and handles loading, error and empty states separately.

### Projects
- **01-basic — DOM with bare hands.** A to-do interface built entirely with
  `createElement`, event delegation and manual re-rendering. Tedious on purpose:
  end with a written note on which parts hurt. Module 06 picks that note back up.
- **02-medium — Asynchronous data view.** Load sensor data from a local JSON
  source, render loading, error and empty states, add debounced search, and abort
  in-flight requests when the input changes.
- **03-final — Tiny reactivity, built from scratch.** A small state object with a
  subscriber mechanism that updates only the affected DOM nodes on change. The
  idea behind React, rebuilt before React shows up.

---

# Block B — Frontend Engineering

## Module 05 — Tooling, Modules, TypeScript

### Theory
Why a build step exists at all: modules in the browser, bundling, transpilation,
minification, tree shaking, source maps. npm in practice: `package.json`,
semantic versioning, the lockfile, `dependencies` versus `devDependencies`,
`node_modules` and where the supply-chain risk sits. Vite as a dev server with
hot module replacement. TypeScript: structural typing, interfaces versus type
aliases, union and literal types, generics in small doses, `unknown` versus
`any`, configuration through `tsconfig.json`. Formatting and linting with ESLint
and Prettier. How the Node ecosystem relates to the Python backend.

### Objectives
Done when a project is set up from zero with Vite, TypeScript and linting; when
the learner reads and fixes a type error without reaching for `any`; and when
they can describe what the build produces from their source.

### Projects
- **01-basic — Setup from scratch.** No template: `package.json`, Vite,
  TypeScript, ESLint, Prettier, scripts, `.gitignore`. Justify every dependency
  in one sentence.
- **02-medium — Typed port.** Port the module 04 medium project to TypeScript,
  including types for API responses and a runtime check at the boundary between
  network and application.
- **03-final — A small type-safe library.** A reusable fetch module with type
  parameters, timeout, retry with backoff and a uniform error object, structured
  for publication and covered by unit tests. Moves into `shared/` and gets reused
  from module 13 onward.

---

## Module 06 — React I: The Component Model

### Theory
Declarative versus imperative, picking up directly from the pain note in module
04. Components as functions, JSX and what it compiles to, props as a one-way
street, composition instead of inheritance. `useState`, state as a snapshot,
batching, why mutation does not work. Lists, keys, and the consequences of using
an index as a key. Conditional rendering. The render cycle: what triggers a
re-render, what the reconciler does. Lifting state up, controlled components.
When a component has grown too large.

### Objectives
Done when the learner can predict which components re-render on a given state
change before running the code; demonstrates the index-key problem on their own
example; and cuts an interface into components with a defensible rationale.

### Projects
- **01-basic — The same interface, again.** Rebuild the module 04 to-do
  interface in React, then compare both versions side by side: lines of code,
  readability, failure modes, what React removed and what it hides.
- **02-medium — Component library.** Reusable building blocks (button, input,
  modal, sortable table, toast) with typed props and a demo page showing every
  variant.
- **03-final — Sensor data explorer.** A filterable, sortable, paginated table
  over a larger synthetic dataset, with state placed deliberately and a written
  justification of where each piece of state lives.

---

## Module 07 — React II: Effects, Data, Routing

### Theory
`useEffect` understood properly: synchronization with the outside world, not a
lifecycle replacement. Dependency array, cleanup, aborting in-flight requests,
the classic infinite loops and how to spot them. When no effect is needed —
derived state belongs in the render function. `useRef`, `useMemo`, `useCallback`
with an honest account of when memoization buys nothing. Custom hooks as an
abstraction tool. Client routing with React Router: routes, params, nested
layouts, protected areas. Forms with validation and error display. Server state
versus client state as the central distinction.

### Objectives
Done when the learner decides, for a given data flow, whether an effect is needed
and can justify it; writes a custom hook encapsulating loading, error and abort;
and ships an app where deep links work.

### Projects
- **01-basic — Effect workshop.** Six small cases: timer, event listener, data
  loading, abort on parameter change, persistence to localStorage, cleanup on
  unmount. For each one: is an effect actually necessary?
- **02-medium — Multi-page application.** Routing with an overview, a detail
  page, search parameters in the URL, a 404 route and nested layouts. The view
  state must be reproducible from a shared link.
- **03-final — Validated form flow.** A multi-step device registration form with
  field validation, server-side errors, draft persistence, and a warning on
  unsaved changes.

---

## Module 08 — UI State, Styling Systems, Data Visualization

### Theory
Why server state follows its own rules: caching, invalidation,
stale-while-revalidate, refetching, optimistic updates. TanStack Query as the
concrete implementation. Client state options compared — context, reducer,
external stores — with the honest answer that most applications need very little.
Prop drilling, and when context is the wrong fix. Utility-first styling with
Tailwind versus handwritten CSS, with the trade-offs both ways. Data
visualization on the web: chart types and their misuses, axis scaling, color
choice including color vision deficiency, gaps in time series, and aggregating
before rendering instead of pushing 50,000 points into the browser.

### Objectives
Done when server and client state are separated in the learner's own code and the
separation is justified; when a chart answers a specific question rather than
decorating data; and when caching behavior is demonstrable in the network tab.

### Projects
- **01-basic — Make caching visible.** The same data view three times: bare
  `fetch`, a hand-rolled cache, TanStack Query. Count requests in the network tab
  and document the differences.
- **02-medium — Analytical dashboard.** Four charts over the synthetic sensor
  dataset (time series, distribution, category comparison, correlation) with a
  date range filter. One sentence per chart: which question does it answer?
- **03-final — Chart critique and redesign.** Analyze four deliberately
  misleading charts (truncated axis, pie chart with twelve slices, wrong
  aggregation, dual axis), rebuild them, and quantify the distortion.

---

# Block C — Python Backend Engineering

## Module 09 — From Raw Server to FastAPI

### Theory
What a web server does, starting at the socket. WSGI and ASGI, sync versus async,
why ASGI won for IO-bound work. Uvicorn and worker processes. FastAPI in
practice: routing, path and query parameters, request bodies, response models,
dependency injection, middleware, lifespan events, router structure for a growing
project. Configuration through environment variables. A project layout that does
not collapse at twenty endpoints. Python backend landscape: FastAPI, Django,
Flask, with a straight answer on when Django is the better call.

### Objectives
Done when the learner can describe what happens between an incoming byte stream
and the function that gets called; when a FastAPI app runs with multiple routers
and dependency injection; and when they can say where `async def` helps and where
it does nothing.

### Projects
- **01-basic — A framework in miniature.** A minimal HTTP server on
  `http.server` that registers routes, parses query parameters, returns JSON and
  distinguishes methods. Roughly 150 lines, followed by: which of these jobs does
  FastAPI take over?
- **02-medium — First FastAPI application.** The same endpoints in FastAPI, with
  Pydantic models, router separation, configuration from the environment, and the
  generated OpenAPI documentation.
- **03-final — Ingestion API for measurements.** A batch upload endpoint for
  sensor readings with validation, a per-record partial failure report,
  idempotency via a client-supplied key, and a health endpoint.

---

## Module 10 — REST API Design and Validation

### Theory
Resource-oriented design: nouns instead of verbs, collections and single
resources, nesting and its limits. Status codes used correctly, uniform error
formats (Problem Details, RFC 9457). Offset versus cursor pagination, with
reasoning. Filtering, sorting, field selection. Versioning. Pydantic v2 in depth:
field validation, custom validators, nested models, separate models for input,
storage and output, and why that separation is a security question rather than a
style preference. OpenAPI as a contract, client type generation. Brief comparison
with GraphQL and gRPC, no deep dive.

### Objectives
Done when a full endpoint design including error cases is produced from a
business requirement before any code is written; when input and output models are
separate; and when cursor pagination behaves correctly over a growing dataset.

### Projects
- **01-basic — API design on paper.** For a described domain, work out every
  endpoint, status code, error format and validation rule as a Markdown
  specification, then formalize it as an OpenAPI document. No implementation.
- **02-medium — Full CRUD with edge cases.** Implement the specification,
  including pagination, filtering, a uniform error handler, and HTTP tests that
  trigger every documented failure case.
- **03-final — Two designs compared.** The same use case built twice: once
  resource-oriented, once as a collection of RPC-style endpoints. Test both
  against identical requirements and write up the verdict.

---

## Module 11 — Persistence and Data Access

### Theory
Relational fundamentals as far as a backend needs them: schema, keys,
relationships, normalization up to third normal form, indexes and what they cost.
Transactions and isolation levels demonstrated through concrete anomalies.
SQLAlchemy 2.0: engine, session, declarative models, relationships, lazy versus
eager loading, and the N+1 problem that module 15 will measure. Migrations with
Alembic, autogenerated migrations and where they mislead. Layered architecture —
router, service, repository, model — and why API models are not database models.
Connection pooling. Time zones: store everything in UTC.

### Objectives
Done when a schema with at least four related tables is designed, migrated and
populated; when the learner reads and judges the SQL an ORM query produces; and
when a schema change ships as a migration without data loss.

### Projects
- **01-basic — Schema and raw SQL.** Design the sensor platform data model,
  create it as SQL DDL, populate it with generated data, and answer ten analytical
  questions in plain SQL. No ORM yet.
- **02-medium — The same queries through the ORM.** Map the models in SQLAlchemy,
  solve the ten queries again, log the generated SQL and compare it with the
  handwritten version. Alembic migrations for two schema changes.
- **03-final — Layered API on a database.** Move the module 10 API onto real
  persistence, with a repository layer, transaction boundaries in the service, a
  fresh test database per run, and one deliberately added index with runtime
  measured before and after.

---

## Module 12 — Authentication, Authorization, Security

### Theory
Authentication versus authorization. Storing passwords correctly: hashing with
bcrypt or Argon2, salting, why SHA-256 is the wrong tool here. Session cookies
versus JWT, the weaknesses of each, expiry and refresh, safe client-side storage
(`HttpOnly`, `Secure`, `SameSite`). Roles and permission checks as dependencies.
OAuth2 and OpenID Connect in overview. Attack classes hands-on: SQL injection,
XSS, CSRF, IDOR, mass assignment, with the OWASP Top 10 as the frame. CORS
finally explained properly, including preflight. Rate limiting, secret
management, logging without personal data.

### Objectives
Done when a login flow runs with hashed passwords and role-based checks; when the
learner executes at least four attacks against a vulnerable sample app and then
fixes them; and when they resolve a CORS error by reasoning rather than guessing.

### Projects
- **01-basic — Attack a vulnerable app.** A deliberately insecure FastAPI
  application is provided. Find five vulnerabilities, exploit them, document them,
  then fix each one and lock the fix in with a test.
- **02-medium — Build the auth system.** Registration, login, password hashing,
  token issuance with expiry, refresh, protected routes, role-based permissions,
  clean logout.
- **03-final — Security review of your own API.** Audit the module 11 API against
  a checklist: per-endpoint authorization, mass assignment, error messages that
  leak internals, rate limiting, security headers. Deliverable is a short review
  report with findings and fixes.

---

# Block D — Integration and Quality

## Module 13 — Full-Stack Integration

### Theory
The seam between the two worlds: the API contract as shared truth, type
generation from OpenAPI, contract drift and how it surfaces. Environments and
configuration on both sides, dev proxy versus production CORS. Error handling
along the whole path: network errors, validation errors, authorization errors,
server errors — each class needs a different user-facing response. Loading
states, optimistic updates and rollback. File uploads. Real time: polling,
server-sent events, WebSockets, with a decision criterion. Monorepo layout for
frontend and backend.

### Objectives
Done when one feature is built through both layers and every error class is
handled distinguishably in the UI; when frontend types are generated from the
backend contract; and when both halves start with a single command.

### Projects
- **01-basic — First feature end to end.** One feature from database column to
  visible element, with generated types and a deliberately introduced contract
  break to show where it gets caught.
- **02-medium — Full CRUD UI with auth.** React frontend against the secured API:
  login, token handling, protected routes, error classes, optimistic updates with
  rollback, CSV upload with progress.
- **03-final — Live view.** Display incoming readings in real time, once via
  polling and once via server-sent events. Compare request count, latency and
  server load, then justify a recommendation.

---

## Module 14 — Testing Across the Stack

### Theory
The test pyramid and the fair criticism of it. Backend with pytest: fixtures,
parametrization, test database, per-test transaction rollback, HTTP tests with
`TestClient` and httpx, mocking external services and when mocking does damage.
Frontend with Vitest and React Testing Library: test behavior, not
implementation; queries by role and text; handling async UI; network mocking with
MSW. End-to-end with Playwright: selector strategy, the real causes of flakiness,
test data setup. Coverage as a tool rather than a target. Test data factories.

### Objectives
Done when the suite reliably catches a deliberately introduced bug; when frontend
tests assert no internal state; and when an end-to-end test walks the main path
reproducibly.

### Projects
- **01-basic — Cover the backend.** Unit and integration tests for the service
  layer with fixtures, parametrization and edge cases. Five given mutations in the
  code must be caught by the suite.
- **02-medium — Frontend tests.** Component and integration tests for the CRUD UI
  using MSW, covering error and loading states as well as keyboard operation.
- **03-final — E2E and a CI gate.** Playwright tests for login, create, filter and
  delete, plus a GitHub Actions workflow that runs all three levels and blocks the
  merge on failure.

---

## Module 15 — Performance, Concurrency, Caching

### Theory
Measure first, optimize second. Where time actually goes: database, network,
rendering. Backend: `async` used correctly, blocking calls inside the event loop
as the most common mistake, thread pools for synchronous libraries, background
tasks and the point where a real worker (Celery, RQ) becomes necessary. Measuring
and fixing the N+1 problem, proving index impact, reading `EXPLAIN`. Caching at
several levels — HTTP headers, application cache, Redis — including the hard part,
invalidation. Frontend: bundle analysis, code splitting, lazy loading, list
virtualization, Core Web Vitals, finding unnecessary re-renders. Load testing with
Locust.

### Objectives
Done when every optimization is backed by a before-and-after measurement; when the
learner spots a blocking call in async code; and when a list of 50,000 rows
scrolls smoothly.

### Projects
- **01-basic — Profiling round.** A deliberately slow application is provided:
  find six bottlenecks, back each with numbers, and rank them by expected payoff.
  No fixes yet.
- **02-medium — Fixes with proof.** Fix the bottlenecks (indexes, eager loading,
  caching, pagination, code splitting, virtualization), measure each change in
  isolation, and produce a results table.
- **03-final — Asynchronous processing.** Move a compute-heavy job (aggregation
  over a large time series) out of the request cycle: trigger the job, poll its
  status, fetch the result, show progress in the UI, and run a Locust load test
  before and after.

---

# Block E — Operations

## Module 16 — Containers, Deployment, CI/CD, Observability

### Theory
Why containers: reproducibility, not virtualization for its own sake. Images and
layers, a Dockerfile for Python and one for a Node-built frontend, multi-stage
builds, image size, non-root users, health checks. Docker Compose for frontend,
backend, PostgreSQL and Redis. Configuration through environment variables,
secrets outside the image. Reverse proxy and serving the frontend build.
Migrations on startup. CI/CD with GitHub Actions: lint, test, build, image push.
Deployment models in overview (VPS, PaaS, container platform) with realistic
costs. Observability: structured JSON logging, correlation IDs across layers,
metrics, health checks, error tracking.

### Objectives
Done when the whole stack starts on a foreign machine with one command; when a
request can be traced through frontend, backend and database by correlation ID in
the logs; and when the pipeline refuses to deploy on a failing test.

### Projects
- **01-basic — Containerize both halves.** Dockerfiles for backend and frontend
  build, multi-stage, non-root, `.dockerignore`, with image size documented before
  and after optimization.
- **02-medium — Compose setup.** A full compose file with database, migration on
  startup, health checks, startup ordering, volumes and `.env.example`. Plus setup
  instructions someone else can follow without asking questions.
- **03-final — Pipeline and observability.** A GitHub Actions workflow with lint,
  test, build and image push; structured logging with a correlation ID through
  both layers; health and metrics endpoints; and one documented incident scenario:
  what do the logs show when the database goes away?

---

# Block F — Capstone

## Module 17 — Capstone: Data Application

### Theory
No lecture script in the usual sense. Instead an architecture document tying the
building blocks of modules 01 to 16 into a decision basis: where each
responsibility sits, which trade-offs were taken deliberately, what scaling would
look like. Plus the specifics of data-heavy web applications: model serving as an
endpoint, separating training from inference, model versioning, the latency budget
for inference, handling large result sets in the browser, export functions.

### Objectives
Done when the application starts from a clean machine via compose, tests are green
at every level, the readme gets a stranger running, and the architecture document
justifies every significant decision.

### Projects
- **01-basic — Design.** Requirements, data model, API contract, component tree,
  architecture diagram and risk list. Before a single line of implementation.
- **02-medium — Build the core.** Auth, CRUD, data ingestion via upload, dashboard
  with charts, filtering, CSV export, tests at every level.
- **03-final — Extend and hand over.** A prediction endpoint that loads a trained
  scikit-learn model and shows a forecast with an uncertainty band in the
  dashboard; a background job for recomputation; compose-based deployment; a
  complete readme, the architecture document, and a short demo script that makes
  the application presentable in a job application.

---

## Order and dependencies

Block A is a prerequisite for everything else. After Block A, Blocks B and C are
independent: take B first for a visible result sooner, take C first to optimize
for the data science goal. Block D needs both. Modules 16 and 17 come last.

**Minimum path** if time runs short: 01, 02, 03, 04, 06, 09, 10, 11, 12, 13, 16,
17. That still produces a complete, presentable application, dropping only the
deep dives on tooling, UI state, testing and performance.
