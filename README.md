<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="basecast-app/docs/assets/logo-dark.svg">
  <img src="basecast-app/docs/assets/logo-light.svg" alt="BaseCast" height="56">
</picture>

**How much of Texas' interconnection queue actually gets built: where, when, and which co-op to call about it.**

[![1st place](https://img.shields.io/badge/%E2%98%85_1st_place-Open_Grid_Data-F4B840?style=flat-square&labelColor=102a17)](https://aitx-base-hackathon-showcase.vercel.app/projects/basecast)
[![Live](https://img.shields.io/badge/live-basecast.pbrenner.com-b2dd79?style=flat-square&labelColor=102a17)](https://basecast.pbrenner.com)
[![Demo video](https://img.shields.io/badge/demo_video-Loom-b2dd79?style=flat-square&labelColor=102a17&logo=loom&logoColor=b2dd79)](https://www.loom.com/share/222a649826174f998aa9b92f530eaca8)

</div>

**Winner of the Open Grid Data track** at the
[Base Power × AITX Talent Hackathon](https://aitx-base-hackathon-showcase.vercel.app/projects/basecast)
(Austin, Sep 25–27, 2026), picked by Base Power's engineering team. A solo entry: the data pipelines, the API
and the app in this repo.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="basecast-app/docs/assets/forecast-dark.webp">
  <img src="basecast-app/docs/assets/forecast-light.webp" alt="Forecast: the summer peak to 2030 in three layers, against ERCOT's official lines">
</picture>

Texas plans its grid around queues that are mostly paper: in January 2026 ERCOT tracked 232.5 GW of large
loads, 3.8% of them approved to energize. BaseCast forecasts how much of the queues gets built, turns that into
a **peak MW forecast by region and year (P10/P50/P90)**, and tells
[Base Power](https://www.basepowercompany.com)'s partnerships team **which co-ops and munis to approach, when,
and with what offer**.

## What's in here

| Folder | What it does | Stack |
|---|---|---|
| [`basecast-airflow/`](basecast-airflow/) | 35 public sources into an immutable lake, typed tables, and the models and marts behind every forecast | Python 3.12, Airflow, Polars, Postgres + PostGIS, BigQuery |
| [`basecast-get-data/`](basecast-get-data/) | The API: one typed endpoint per resource, every answer with its provenance; owns the data contract | FastAPI, Pydantic 2, Polars, OpenAPI 3.1 |
| [`basecast-app/`](basecast-app/) | The product: county map, peak forecast, backtest, and the ranked co-op accounts | Next.js 16, React 19, Tailwind 4, TanStack Query, MapLibre |

Each folder has its own README, toolchain and lockfile. They started as three repos and moved here on
2026-11-19 with their full history.

## How it fits

```mermaid
flowchart TB
  src["Public data: ERCOT · PUCT · EIA · Census · BLS · NOAA · TCEQ · …"]

  subgraph airflow["basecast-airflow"]
    direction LR
    raw["raw<br/>immutable snapshots"] --> process["process<br/>typed tables"] --> marts["marts<br/>models + golden checks"]
  end

  subgraph getdata["basecast-get-data"]
    direction LR
    api["FastAPI<br/>one typed endpoint per resource"] --> meta["{data, meta}<br/>provenance + caveats"]
  end

  subgraph app["basecast-app"]
    direction LR
    bff["BFF route handlers<br/>token stays on the server"] --> ui["Next.js screens<br/>on Vercel"]
  end

  src --> airflow
  airflow -- "Postgres · GCS · BigQuery" --> getdata
  getdata -- "openapi.json → TS client" --> app
```

> [!NOTE]
> **The cloud side was shut down on 2026-09-29.** After the hackathon the GCP project (the Airflow VM, Cloud
> Run, Cloud SQL, the lake's bucket, BigQuery) was deleted to bring the bill to zero. The live demo serves a
> snapshot of the API's answers recorded that day (`basecast-app/snapshot/`). The pipelines and the API still
> run locally, against a lake rebuilt from the public sources.

## Author

Pablo Brenner · [LinkedIn](https://www.linkedin.com/in/brennerpablo/) · [pbrenner.com](https://www.pbrenner.com)
