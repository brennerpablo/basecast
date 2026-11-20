# CLAUDE.md — basecast

Monorepo since 2026-11-19: the three hackathon repos, one folder each, with their full history.

| Folder | Front | Stack |
|---|---|---|
| `basecast-airflow/` | A. Data pipelines: public sources → lake → typed tables → marts | Python 3.12, uv, Airflow, Polars |
| `basecast-get-data/` | C. API and data contract (owns `openapi.json`) | Python 3.12, uv, FastAPI |
| `basecast-app/` | B. Frontend, the live static demo on Vercel | Next.js, npm |

Each folder keeps its own `CLAUDE.md`, toolchain and lockfile: run commands from inside the folder.
Cross-folder decisions are logged in `basecast-get-data/docs/decisions.md`. Only `basecast-app` is
deployed (Vercel, Root Directory `basecast-app`); the GCP services behind the other two were deleted on
2026-09-29, so their `.github/workflows/` are a record and never run (GitHub only reads the root's).
