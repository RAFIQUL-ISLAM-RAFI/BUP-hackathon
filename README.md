# GridWise — LLM-Assisted Smart Campus Energy Optimization

**BUP CSE Fest 2026 Hackathon · Online Preliminary**

An HTTP API service that interprets natural-language operator notes using an LLM,
converts them into deterministic structured directives, applies them to a 24-hour
energy optimization problem, and returns a valid, low-cost schedule.

---

## 📋 Table of Contents

1. [Quick Start (Docker)](#1-quick-start-docker)
2. [Quick Start (Local)](#2-quick-start-local)
3. [API Reference](#3-api-reference)
4. [Sample Request & Response](#4-sample-request--response)
5. [Public Sample Test](#5-public-sample-test)
6. [Architecture](#6-architecture)
7. [LLM Role & Guardrails](#7-llm-role--guardrails)
8. [Environment Variables](#8-environment-variables)
9. [Dependencies](#9-dependencies)
10. [Model / Provider Disclosure](#10-model--provider-disclosure)
11. [Optimizer / Solver](#11-optimizer--solver)
12. [Known Limitations](#12-known-limitations)
13. [Secret Handling](#13-secret-handling)
14. [Credits & Tools](#14-credits--tools)

---

## 1. Quick Start (Docker)

Pull the pre-built image and run it. No local Python setup required.

```bash
# Pull image (replace with actual registry + tag)
docker pull <your-dockerhub-username>/gridwise:1.0.0

# Run with your Gemini API key
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=AIza-your-real-key-here \
  -e GEMINI_MODEL=gemini-3.5-flash-lite \
  <your-dockerhub-username>/gridwise:1.0.0
```

**Verify health:**

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

**Exposed port:** `8000` (container binds to `0.0.0.0`)

**Required environment variables:**

| Name | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (primary) | Google Gemini API key — free tier works |
| `GEMINI_MODEL` | No | Defaults to `gemini-3.5-flash-lite` |
| `OPENAI_API_KEY` | No | Optional paid fallback |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |
| `LOG_LEVEL` | No | `INFO` (default) or `DEBUG` |

**⚠️ Do NOT bake secrets into the image. Always pass at runtime via `-e`.**

---

## 2. Quick Start (Local)

From a clean environment (Python 3.10+ required):

```bash
# 1. Clone repository
git clone <your-repo-url>
cd gridwise

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
#    Copy the example and fill in your real Gemini API key
copy .env.example .env       # Windows
cp .env.example .env         # macOS / Linux

# 5. Run the service
python gridwise.py
```

**Expected startup output:**

```
============================================================
 GridWise LLM-Assisted Energy Optimizer
============================================================
 Starting server on http://0.0.0.0:8000
 Interactive docs: http://localhost:8000/docs
 Health:           http://localhost:8000/health
 Run CLI tests:    python gridwise.py --test
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Run built-in test suite (no server needed):**

```bash
python gridwise.py --test
```

**Or open interactive Swagger UI:**

```
http://localhost:8000/docs
```

---

## 3. API Reference

### `GET /health`

Readiness probe for the judging harness.

**Request:**

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**

```json
{"status": "ok"}
```

Returns 200 within 60 seconds of service start.

---

### `POST /optimize-energy`

Accepts one scenario (24 hours + battery + 1–3 operator notes) and returns
the LLM interpretation + optimal 24-hour schedule.

**Request headers:**

```
Content-Type: application/json
```

**Request body schema:**

```jsonc
{
  "scenario_id": "string",
  "operator_notes": ["string", ...],       // 1–3 natural-language notes
  "hours": [                                // exactly 24 entries, hours 0..23
    {
      "hour": 0,
      "demand_kwh": 180,
      "solar_kwh": 0,
      "tariff_bdt_per_kwh": 7
    }
    // ... 23 more
  ],
  "battery": {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100
  }
}
```

**Response (200 OK):** full JSON with `directive_interpretation`, `hourly_plan`,
`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`, `plan_summary`.

**Response codes:**

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Malformed JSON / invalid request schema |
| 422 | Semantically invalid but well-formed request |
| 500 | Controlled internal error (no secrets leaked) |

---

## 4. Sample Request & Response

**Sample request** (save as `sample.json`):

```json
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    {"hour": 0,  "demand_kwh": 180, "solar_kwh": 0,   "tariff_bdt_per_kwh": 7},
    {"hour": 1,  "demand_kwh": 170, "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
    {"hour": 2,  "demand_kwh": 160, "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
    {"hour": 3,  "demand_kwh": 160, "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
    {"hour": 4,  "demand_kwh": 170, "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
    {"hour": 5,  "demand_kwh": 190, "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
    {"hour": 6,  "demand_kwh": 220, "solar_kwh": 10,  "tariff_bdt_per_kwh": 8},
    {"hour": 7,  "demand_kwh": 260, "solar_kwh": 40,  "tariff_bdt_per_kwh": 10},
    {"hour": 8,  "demand_kwh": 300, "solar_kwh": 100, "tariff_bdt_per_kwh": 12},
    {"hour": 9,  "demand_kwh": 330, "solar_kwh": 180, "tariff_bdt_per_kwh": 14},
    {"hour": 10, "demand_kwh": 350, "solar_kwh": 260, "tariff_bdt_per_kwh": 16},
    {"hour": 11, "demand_kwh": 360, "solar_kwh": 320, "tariff_bdt_per_kwh": 16},
    {"hour": 12, "demand_kwh": 370, "solar_kwh": 360, "tariff_bdt_per_kwh": 15},
    {"hour": 13, "demand_kwh": 360, "solar_kwh": 340, "tariff_bdt_per_kwh": 14},
    {"hour": 14, "demand_kwh": 340, "solar_kwh": 280, "tariff_bdt_per_kwh": 13},
    {"hour": 15, "demand_kwh": 330, "solar_kwh": 180, "tariff_bdt_per_kwh": 14},
    {"hour": 16, "demand_kwh": 340, "solar_kwh": 90,  "tariff_bdt_per_kwh": 18},
    {"hour": 17, "demand_kwh": 370, "solar_kwh": 20,  "tariff_bdt_per_kwh": 22},
    {"hour": 18, "demand_kwh": 410, "solar_kwh": 0,   "tariff_bdt_per_kwh": 28},
    {"hour": 19, "demand_kwh": 430, "solar_kwh": 0,   "tariff_bdt_per_kwh": 30},
    {"hour": 20, "demand_kwh": 410, "solar_kwh": 0,   "tariff_bdt_per_kwh": 26},
    {"hour": 21, "demand_kwh": 350, "solar_kwh": 0,   "tariff_bdt_per_kwh": 18},
    {"hour": 22, "demand_kwh": 270, "solar_kwh": 0,   "tariff_bdt_per_kwh": 10},
    {"hour": 23, "demand_kwh": 210, "solar_kwh": 0,   "tariff_bdt_per_kwh": 7}
  ],
  "battery": {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100
  }
}
```

**Curl command:**

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample.json
```

**Sample response (truncated):**

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Solar availability reduced to 25% during panel cleaning."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "Registration deadline does not affect today's energy schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90, "solar_used_kwh": 0, "battery_action": "idle", "battery_kwh": 0, "battery_energy_after_kwh": 200},
    /* ... 23 more entries ... */
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365,
  "peak_grid_kwh": 175,
  "plan_summary": "Applied 1 directive: solar_reduction. Optimized grid cost."
}
```

---

## 5. Public Sample Test

Run all 10 public sample cases from the BUP problem pack.

**Requirements:** Service running on `http://localhost:8000`, public sample JSON present.

```bash
# Set paths
export GRIDWISE_URL=http://localhost:8000
export SAMPLES_JSON=BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json

# Run test
python test_public.py
```

**Expected output:**

```
[OK]   SAMPLE-01: cost=28735 ref=38365 ratio=0.749 dir=True t=3.2s
[OK]   SAMPLE-02: cost=24804 ref=42885 ratio=0.578 dir=True t=2.1s
...
============================================================
 Passed: 10 / 10
 Failed: 0
 Avg time: 3.4s
============================================================
```

**Pass criteria:**
- Every note produces one interpretation in `note_index` order
- Directive type, `applies`, and `structured_adjustment` match reference semantics
- Recalculated cost ≤ reference cost × 1.02 (numerical tolerance)
- Response latency < 30 s

---

## 6. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Service                          │
│                                                              │
│  GET  /health  ──►  {"status": "ok"}                        │
│                                                              │
│  POST /optimize-energy                                       │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Request  │─►│   LLM    │─►│Guardrails│─►│    LP    │    │
│  │Validator │  │Interpreter│  │Validator │  │Optimizer │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                   │          │
│                                                   ▼          │
│                                          ┌──────────────┐   │
│                                          │    Final     │   │
│                                          │  Validator   │   │
│                                          └──────┬───────┘   │
│                                                 │            │
│                                                 ▼            │
│                                          JSON Response       │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline stages:**

| Stage | Responsibility | File location |
|---|---|---|
| Request validator | Pydantic schema; rejects malformed input with 400/422 | `gridwise.py` (schemas section) |
| LLM interpreter | Converts each note into exactly one structured directive | `gridwise.py` (`_call_gemini`, `_call_openai`, `_rule_based_interpret`) |
| Guardrail validator | Deterministic — allowed types, hour format, numeric ranges, no_op shape | `gridwise.py` (`validate_and_normalize`) |
| LP optimizer | SciPy HiGHS linear program — 120 continuous variables, minimizes grid cost | `gridwise.py` (`solve_plan`) |
| Final validator | Replays plan against every rule before returning | `gridwise.py` (`validate_plan`) |

---

## 7. LLM Role & Guardrails

### LLM role (mandatory)

The LLM **must** sit on the interpretation path. It is called for every
operator note and produces one structured directive per note. The LLM never
performs math directly.

**Prompt engineering highlights:**

- Temperature = 0 (deterministic)
- `response_mime_type="application/json"` (enforced structured output)
- Explicit time convention rules (`1 PM to 3 PM → [13, 14]`)
- Explicit solar factor rule (`80% reduction → factor 0.2`)
- Explicit `no_op` rules

### Deterministic guardrails

After the LLM response is returned, it is **untrusted** until it passes these
checks (all implemented in `validate_and_normalize`):

- Exactly one entry per note, in `note_index` order
- `directive_type` ∈ {`solar_reduction`, `minimum_battery_reserve`,
  `no_charge_window`, `no_discharge_window`, `max_grid_window`, `no_op`}
- `no_op` → `applies = false`, `structured_adjustment = null`
- Every other directive → `applies = true`
- `hours` = unique integers 0–23, ascending
- `solar_reduction.factor` ∈ [0, 1]
- `minimum_battery_reserve.minimum_energy_kwh` ≥ 0
- `max_grid_window.max_grid_kwh` ≥ 0

If validation fails, the fallback is to mark all notes as `no_op` — the
service stays alive and produces a valid base schedule.

### Emergency fallback

If **both** Gemini and OpenAI are unreachable, a rule-based regex interpreter
(`_rule_based_interpret`) is used. It handles common phrasings and keeps the
service responsive; it is not as paraphrase-robust as the LLM path.

---

## 8. Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (recommended) | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Gemini model name |
| `OPENAI_API_KEY` | No | — | Optional paid fallback |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model name |
| `LOG_LEVEL` | No | `INFO` | `INFO` or `DEBUG` |

**How to obtain a Gemini key (free tier):**

1. Visit https://aistudio.google.com/apikey
2. Click **Create API Key**
3. Copy the key (starts with `AIza…`)
4. Save to `.env` as `GEMINI_API_KEY=…`

Do **not** commit `.env` to the repository. `.gitignore` excludes it.

---

## 9. Dependencies

**Python 3.10+ required.**

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | ≥ 0.115 | Web framework |
| `uvicorn[standard]` | ≥ 0.32 | ASGI server |
| `pydantic` | ≥ 2.9 | Schema validation |
| `scipy` | ≥ 1.14 | LP solver (HiGHS) |
| `numpy` | ≥ 2.1 | Numeric arrays |
| `google-genai` | ≥ 1.0 | Gemini SDK (new API) |
| `openai` | ≥ 1.52 | Optional OpenAI fallback |
| `python-dotenv` | ≥ 1.0 | `.env` loading |
| `tenacity` | ≥ 9.0 | Retry helpers |
| `requests` | ≥ 2.32 | Public-sample test client |

Install everything in one command:

```bash
pip install fastapi "uvicorn[standard]" pydantic scipy numpy google-genai openai python-dotenv tenacity requests
```

---

## 10. Model / Provider Disclosure

| Provider | Role | Model | Cost |
|---|---|---|---|
| Google Gemini | Primary LLM | `gemini-3.5-flash-lite` | Free tier |
| OpenAI | Optional fallback | `gpt-4o-mini` | Paid ($5 minimum) |
| Rule-based regex | Emergency fallback | — | Free |

**Why Gemini as primary:** free tier with generous rate limits
(15 req/min, 1M tokens/day), fast response, and stable `flash-lite` throughput.

**Why OpenAI as fallback:** optional paid backup for teams that prefer OpenAI
or need higher throughput.

---

## 11. Optimizer / Solver

**Solver:** SciPy `linprog` with `method="highs"` (HiGHS — open-source LP).

**Variables (per hour `h`):**

| Variable | Meaning | Bounds |
|---|---|---|
| `g[h]` | Grid energy purchased | `[0, max_grid_window]` (if active) |
| `s[h]` | Solar energy used | `[0, effective_solar[h]]` |
| `c[h]` | Battery charge | `[0, max_charge_kwh_per_hour]` (or 0 in no-charge window) |
| `d[h]` | Battery discharge | `[0, max_discharge_kwh_per_hour]` (or 0 in no-discharge window) |
| `e[h]` | Battery energy after hour `h` | `[active_min_energy[h], capacity_kwh]` |

**Constraints:**

1. Energy balance: `g[h] + s[h] + d[h] = demand[h] + c[h]`
2. Battery dynamics: `e[h] = e[h-1] + c[h] - d[h]`
3. End-of-day neutrality: `e[23] = initial_energy_kwh`
4. Directive bounds: `no_charge`, `no_discharge`, `max_grid_window`,
   `minimum_battery_reserve` applied as bounds
5. Effective solar reduced by `solar_reduction` factor

**Objective:** `minimize Σ g[h] × tariff_bdt_per_kwh[h]`

**Why LP is sufficient:** all constraints and the objective are linear, so
HiGHS finds the global optimum in milliseconds. No MILP needed.

**Numerical tolerance:** 0.01 kWh / 0.01 BDT (per problem statement).

---

## 12. Known Limitations

1. **Single-objective optimization** — cost only. Peak-grid minimization is
   computed but not a secondary objective.
2. **No grid export** — per problem statement, unused solar is curtailed,
   grid export is out of scope.
3. **Simultaneous charge+discharge prevention** — LP may produce both at
   equal cost; a post-solve cleanup resolves this by taking the larger
   magnitude.
4. **Rule-based fallback is limited** — if both LLM providers are down, the
   regex interpreter handles common phrasings but may miss unusual
   paraphrases. The LLM path is the primary interpretation mechanism.
5. **First-call latency** — the initial LLM request may take ~10 s due to
   cold-start on Gemini. Subsequent calls are ~2 s. Public-sample test
   averages < 5 s.
6. **Single-region deployment** — not horizontally scaled; the service
   relies on the platform's restart policy for high availability.

---

## 13. Secret Handling

- **No secrets in repository** — `.env` is listed in `.gitignore`.
- **No secrets in Docker image** — image passes env vars at runtime via `-e`.
- **No secrets in logs** — the logger truncates error messages to 150–250
  chars; the FastAPI 500 handler never echoes stack traces to the client.
- **No secrets in responses** — the response model is fixed; no free-form
  fields leak provider keys or prompts.
- **`.env.example`** ships with placeholder values only.

To rotate the Gemini key:

1. Generate a new key at https://aistudio.google.com/apikey
2. Update `.env` (local) or the platform's env-var settings (deployed)
3. Restart the service

---

## 14. Credits & Tools

**Built by:** `<Your Team Name>`

**Third-party libraries & tools:**

- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [Pydantic](https://docs.pydantic.dev/) — schema validation
- [SciPy](https://scipy.org/) / [HiGHS](https://highs.dev/) — LP solver
- [Google Gemini API](https://ai.google.dev/) — LLM interpretation
- [OpenAI API](https://platform.openai.com/) — optional fallback LLM
- [Uvicorn](https://www.uvicorn.org/) — ASGI server

**Assistance:** AI coding assistants were used for boilerplate generation and
prompt engineering. Core architecture, optimization model, guardrails, and
validation logic were implemented by the team.

**Reference documents:**

- BUP CSE Fest 2026 Preliminary Problem Statement — GridWise LLM
- BUP CSE Fest 2026 Participant Guide & Evaluation Rubric
- BUP CSE Fest 2026 Public Sample Cases JSON (v2.0)

---

## 📞 Contact

For issues running this service, contact: `<your-team-email>`

---

**Version:** 1.0.0
**Last updated:** 2026-09-18
