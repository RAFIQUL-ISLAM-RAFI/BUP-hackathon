"""
GridWise â€” LLM-Assisted Smart Campus Energy Optimization
Complete single-file system for BUP CSE Fest 2026 Hackathon.

Run:
    pip install fastapi uvicorn[standard] pydantic scipy numpy openai google-generativeai python-dotenv tenacity requests
    python gridwise.py

Endpoints:
    GET  /health
    POST /optimize-energy

Or open http://localhost:8000/docs for interactive Swagger UI.
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import re
import json
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Literal

import numpy as np
from scipy.optimize import linprog
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gridwise")


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
DirectiveType = Literal[
    "solar_reduction", "minimum_battery_reserve", "no_charge_window",
    "no_discharge_window", "max_grid_window", "no_op"
]
BatteryAction = Literal["charge", "discharge", "idle"]


class HourInput(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class Battery(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class ScenarioRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourInput] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("hours")
    @classmethod
    def hours_complete(cls, v):
        if sorted(h.hour for h in v) != list(range(24)):
            raise ValueError("hours must contain exactly 0..23")
        return sorted(v, key=lambda x: x.hour)

    @field_validator("operator_notes")
    @classmethod
    def notes_nonempty(cls, v):
        if any(not n.strip() for n in v):
            raise ValueError("operator_notes must be non-empty strings")
        return v


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict] = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# ============================================================
# LLM INTERPRETER
# ============================================================
SYSTEM_PROMPT = """You are an energy-schedule directive extractor for a smart campus.

Given ONE operator note and battery context, return EXACTLY ONE directive as compact JSON.

Directive types and their structured_adjustment shapes:
1. solar_reduction: {"hours": [int,...], "factor": float 0..1}
   - factor = usable fraction REMAINING after the event.
   - "80% reduction" -> factor = 0.2 ; "20% of normal" -> factor = 0.2
   - "roughly 25% of forecast" -> 0.25 ; "about half" -> 0.5
2. minimum_battery_reserve: {"hours": [...], "minimum_energy_kwh": float}
   - If note uses a percent of battery capacity, convert using capacity_kwh.
3. no_charge_window: {"hours": [...]}
4. no_discharge_window: {"hours": [...]}
5. max_grid_window: {"hours": [...], "max_grid_kwh": float}
6. no_op: null  -> use ONLY if note is irrelevant to today's 24h energy schedule.

TIME CONVENTION (CRITICAL):
- Start hour INCLUDED, end hour EXCLUDED.
- "1 PM to 3 PM" -> [13, 14]
- "noon until 2 PM" -> [12, 13]
- "from 6 PM until 9 PM" -> [18, 19, 20]
- "from 2 AM until 5 AM" -> [2, 3, 4]
- Use 24-hour integers 0..23, unique, ascending.

no_op RULES: applies=false, structured_adjustment=null.
NON-no_op RULES: applies=true.

Output format (JSON ONLY, no markdown):
{"applies": bool, "directive_type": "string", "structured_adjustment": {...}|null, "explanation": "short reason"}
"""

USER_TEMPLATE = """Battery context:
- capacity_kwh: {cap}
- initial_energy_kwh: {init}
- minimum_energy_kwh: {min_e}

Operator note:
\"\"\"{note}\"\"\"

Return JSON only."""


def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("sk-your"):
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        log.warning("OpenAI client init failed: %s", e)
        return None


def _call_openai(note: str, battery: Dict[str, Any]) -> Optional[dict]:
    client = _openai_client()
    if client is None:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")        
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    cap=battery["capacity_kwh"],
                    init=battery["initial_energy_kwh"],
                    min_e=battery["minimum_energy_kwh"],
                    note=note,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=10,
            max_retries=0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.warning("OpenAI call failed: %s", str(e)[:150])
        return None


def _call_gemini(note: str, battery: Dict[str, Any]) -> Optional[dict]:
    key = os.getenv("GEMINI_API_KEY")
    if not key or key.startswith("your-"):
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=key,
            http_options={"timeout": 15_000},  # 60s timeout in ms
        )
        prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
            cap=battery["capacity_kwh"],
            init=battery["initial_energy_kwh"],
            min_e=battery["minimum_energy_kwh"],
            note=note,
        )
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return json.loads(resp.text)
    except Exception as e:
        log.warning("Gemini call failed: %s", str(e)[:250])
        return None


def _parse_clock(token: str) -> Optional[int]:
    t = token.strip().lower()
    if t == "noon":
        return 12
    if t == "midnight":
        return 0
    m = re.match(r"(\d{1,2})(?::\d{2})?\s*(am|pm)?", t)
    if not m:
        return None
    h = int(m.group(1))
    ap = m.group(2)
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h


def _extract_window(text: str) -> Optional[List[int]]:
    tl = text.lower()
    time_pat = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)"
    m = re.search(
        rf"(?:from|between|during)\s+{time_pat}\s+(?:to|until|through|till|-)\s+{time_pat}",
        tl
    )
    if not m:
        m = re.search(
            rf"{time_pat}\s+(?:to|until|through|till|-)\s+{time_pat}",
            tl
        )
    if not m:
        return None
    s = _parse_clock(m.group(1))
    e = _parse_clock(m.group(2))
    if s is None or e is None or e <= s:
        return None
    return list(range(s, e))

def _rule_based_interpret(note: str, battery: Dict[str, Any]) -> dict:
    text = note.lower()
    hours = _extract_window(note)

    energy_kw = ["solar", "pv", "panel", "battery", "charge", "discharge", "grid",
                 "reserve", "import", "feeder", "transformer", "kwh", "energy"]
    if not any(k in text for k in energy_kw):
        return {"applies": False, "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Note is unrelated to today's energy schedule."}

    # solar_reduction
    if any(k in text for k in ["solar", "pv", "panel", "photovoltaic"]) and hours:
        factor = None
        m = re.search(r"(\d+)\s*%\s*reduction", text)
        if m:
            factor = 1 - int(m.group(1)) / 100
        else:
            m = re.search(r"(\d+)\s*%\s*(?:of|remaining|output|normal|forecast)", text)
            if m:
                factor = int(m.group(1)) / 100
            elif "half" in text:
                factor = 0.5
            elif "one-?fifth" in text:
                factor = 0.2
            elif "quarter" in text:
                factor = 0.25
        if factor is not None:
            return {"applies": True, "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": hours, "factor": round(factor, 4)},
                    "explanation": "Solar availability reduced."}

    # no_charge_window
    if any(k in text for k in ["no charge", "charging is disabled", "charger",
                                "cannot charge", "charging unavailable",
                                "charging circuit", "not charge",
                                "charging is unavailable"]) and hours:
        return {"applies": True, "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": "Charging disabled in window."}

    # no_discharge_window
    if any(k in text for k in ["no discharge", "not discharge",
                                "discharge is disabled", "do not discharge",
                                "discharging unavailable", "must not discharge"]) and hours:
        return {"applies": True, "directive_type": "no_discharge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": "Discharge disabled in window."}

    # max_grid_window
    if any(k in text for k in ["grid import", "grid intake", "feeder",
                                "transformer", "grid cap",
                                "import must not", "import may not"]) and hours:
        m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text)
        cap = float(m.group(1)) if m else None
        if cap is not None:
            return {"applies": True, "directive_type": "max_grid_window",
                    "structured_adjustment": {"hours": hours, "max_grid_kwh": cap},
                    "explanation": "Grid import capped."}

    # minimum_battery_reserve
    if any(k in text for k in ["reserve", "at least", "keep",
                                "remain in the battery", "remain in battery"]) and hours:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            e = float(m.group(1)) / 100 * battery["capacity_kwh"]
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text)
            e = float(m.group(1)) if m else None
        if e is not None:
            return {"applies": True, "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {"hours": hours, "minimum_energy_kwh": e},
                    "explanation": "Battery reserve enforced."}

    return {"applies": False, "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Could not map note to a supported directive."}


@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=0.4, max=2))
def _interpret_one(note: str, battery: Dict[str, Any]) -> dict:
    callers = []

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # Prefer Gemini (free tier)
    if gemini_key and not gemini_key.startswith("your-"):
        callers.append(_call_gemini)
    # Then OpenAI (paid)
    if openai_key and not openai_key.startswith("sk-your"):
        callers.append(_call_openai)

    if not callers:
        log.warning("No LLM API key configured; using rule-based fallback")
        return _rule_based_interpret(note, battery)

    for caller in callers:
        out = caller(note, battery)
        if out and isinstance(out, dict) and "directive_type" in out:
            return out

    log.info("All LLM providers failed; using rule-based fallback")
    return _rule_based_interpret(note, battery)


def interpret_notes(notes: List[str], battery: Dict[str, Any]) -> List[dict]:
    results = []
    for i, note in enumerate(notes):
        try:
            d = _interpret_one(note, battery)
        except Exception as e:
            log.exception("LLM interpretation crashed for note %d: %s", i, e)
            d = _rule_based_interpret(note, battery)
        d["note_index"] = i
        results.append(d)
    return results


# ============================================================
# GUARDRAILS
# ============================================================
ALLOWED_TYPES = {
    "solar_reduction", "minimum_battery_reserve",
    "no_charge_window", "no_discharge_window",
    "max_grid_window", "no_op",
}


def _normalize_hours(hours) -> List[int]:
    if not isinstance(hours, list):
        raise ValueError("hours must be a list")
    out = []
    for h in hours:
        if isinstance(h, bool) or not isinstance(h, (int, float)):
            raise ValueError(f"invalid hour value: {h!r}")
        hi = int(h)
        if hi < 0 or hi > 23:
            raise ValueError(f"hour out of range: {hi}")
        out.append(hi)
    return sorted(set(out))


def _validate_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    dt = e.get("directive_type")
    if dt not in ALLOWED_TYPES:
        raise ValueError(f"unsupported directive_type: {dt!r}")

    if dt == "no_op":
        return {
            "note_index": e["note_index"],
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": str(e.get("explanation", ""))[:500],
        }

    if not e.get("applies", True):
        e["applies"] = True

    adj = e.get("structured_adjustment")
    if not isinstance(adj, dict):
        raise ValueError("structured_adjustment must be an object for non-no_op")

    hours = _normalize_hours(adj.get("hours", []))
    if not hours:
        raise ValueError("hours must be non-empty")

    clean = {"hours": hours}

    if dt == "solar_reduction":
        f = adj.get("factor")
        if not isinstance(f, (int, float)) or f < 0 or f > 1:
            raise ValueError(f"factor must be in [0,1], got {f!r}")
        clean["factor"] = float(f)
    elif dt == "minimum_battery_reserve":
        v = adj.get("minimum_energy_kwh")
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"minimum_energy_kwh invalid: {v!r}")
        clean["minimum_energy_kwh"] = float(v)
    elif dt == "max_grid_window":
        v = adj.get("max_grid_kwh")
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"max_grid_kwh invalid: {v!r}")
        clean["max_grid_kwh"] = float(v)

    return {
        "note_index": e["note_index"],
        "applies": True,
        "directive_type": dt,
        "structured_adjustment": clean,
        "explanation": str(e.get("explanation", ""))[:500],
    }


def validate_and_normalize(entries: List[Dict[str, Any]], n_notes: int) -> List[Dict[str, Any]]:
    if not isinstance(entries, list) or len(entries) != n_notes:
        raise ValueError(f"expected {n_notes} entries, got {len(entries) if isinstance(entries, list) else 'invalid'}")
    by_index = {}
    for e in entries:
        idx = e.get("note_index")
        if not isinstance(idx, int) or idx < 0 or idx >= n_notes:
            raise ValueError(f"invalid note_index: {idx!r}")
        if idx in by_index:
            raise ValueError(f"duplicate note_index {idx}")
        by_index[idx] = e
    ordered = [by_index[i] for i in range(n_notes)]
    return [_validate_entry(e) for e in ordered]


# ============================================================
# OPTIMIZER (LP)
# ============================================================
def _apply_directives(hours_in, battery, interpretations):
    n = 24
    eff = [hours_in[h]["solar_kwh"] for h in range(n)]
    min_e = [battery["minimum_energy_kwh"]] * n
    no_c, no_d = set(), set()
    maxg = [None] * n

    for e in interpretations:
        dt = e["directive_type"]
        if dt == "no_op":
            continue
        adj = e["structured_adjustment"]
        if dt == "solar_reduction":
            for h in adj["hours"]:
                eff[h] = eff[h] * adj["factor"]
        elif dt == "minimum_battery_reserve":
            for h in adj["hours"]:
                min_e[h] = max(min_e[h], adj["minimum_energy_kwh"])
        elif dt == "no_charge_window":
            no_c.update(adj["hours"])
        elif dt == "no_discharge_window":
            no_d.update(adj["hours"])
        elif dt == "max_grid_window":
            for h in adj["hours"]:
                cap = adj["max_grid_kwh"]
                maxg[h] = cap if maxg[h] is None else min(maxg[h], cap)

    return eff, min_e, no_c, no_d, maxg


def solve_plan(hours_in, battery, interpretations):
    n = 24
    eff, min_e, no_c, no_d, maxg = _apply_directives(hours_in, battery, interpretations)

    def gi(h): return h
    def si(h): return n + h
    def ci(h): return 2 * n + h
    def di(h): return 3 * n + h
    def ei(h): return 4 * n + h

    nv = 5 * n

    c = np.zeros(nv)
    for h in range(n):
        c[gi(h)] = hours_in[h]["tariff_bdt_per_kwh"]

    A_eq_rows, b_eq = [], []

    # Energy balance
    for h in range(n):
        row = np.zeros(nv)
        row[gi(h)] = 1.0
        row[si(h)] = 1.0
        row[di(h)] = 1.0
        row[ci(h)] = -1.0
        A_eq_rows.append(row)
        b_eq.append(hours_in[h]["demand_kwh"])

    # Battery dynamics
    for h in range(n):
        row = np.zeros(nv)
        row[ei(h)] = 1.0
        row[ci(h)] -= 1.0
        row[di(h)] += 1.0
        if h == 0:
            b_eq.append(battery["initial_energy_kwh"])
        else:
            row[ei(h - 1)] = -1.0
            b_eq.append(0.0)
        A_eq_rows.append(row)

    # End-of-day neutrality
    row = np.zeros(nv)
    row[ei(n - 1)] = 1.0
    A_eq_rows.append(row)
    b_eq.append(battery["initial_energy_kwh"])

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq)

    bounds = []
    for h in range(n):
        upper_g = maxg[h] if maxg[h] is not None else None
        bounds.append((0.0, upper_g))
    for h in range(n):
        bounds.append((0.0, max(0.0, eff[h])))
    for h in range(n):
        bounds.append((0.0, 0.0 if h in no_c else battery["max_charge_kwh_per_hour"]))
    for h in range(n):
        bounds.append((0.0, 0.0 if h in no_d else battery["max_discharge_kwh_per_hour"]))
    for h in range(n):
        lo = max(min_e[h], 0.0)
        hi = battery["capacity_kwh"]
        if lo > hi:
            lo = hi
        bounds.append((lo, hi))

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP infeasible: {res.message}")

    x = res.x
    plan = []
    for h in range(n):
        g = max(0.0, x[gi(h)])
        s = max(0.0, x[si(h)])
        cc = max(0.0, x[ci(h)])
        dd = max(0.0, x[di(h)])
        e = x[ei(h)]

        if cc > 1e-4 and dd > 1e-4:
            if cc >= dd:
                cc -= dd
                dd = 0.0
            else:
                dd -= cc
                cc = 0.0

        if cc > 1e-4:
            action, kwh = "charge", cc
        elif dd > 1e-4:
            action, kwh = "discharge", dd
        else:
            action, kwh = "idle", 0.0

        if abs(g) < 1e-6: g = 0.0
        if abs(s) < 1e-6: s = 0.0
        if kwh < 1e-6:
            kwh = 0.0
            action = "idle"

        plan.append({
            "hour": h,
            "grid_kwh": round(g, 6),
            "solar_used_kwh": round(s, 6),
            "battery_action": action,
            "battery_kwh": round(kwh, 6),
            "battery_energy_after_kwh": round(e, 6),
        })

    return plan


# ============================================================
# FINAL VALIDATOR
# ============================================================
def validate_plan(hours_in, battery, interpretations, plan):
    TOL = 0.01
    assert len(plan) == 24
    assert sorted(p["hour"] for p in plan) == list(range(24))

    eff, min_e, no_c, no_d, maxg = _apply_directives(hours_in, battery, interpretations)
    prev_e = battery["initial_energy_kwh"]

    for p in plan:
        h = p["hour"]
        g = p["grid_kwh"]
        s = p["solar_used_kwh"]
        act = p["battery_action"]
        kwh = p["battery_kwh"]
        e_after = p["battery_energy_after_kwh"]

        assert g >= -TOL
        assert s >= -TOL
        assert s <= eff[h] + TOL
        assert kwh >= -TOL
        assert act in ("charge", "discharge", "idle")

        if act == "idle":
            assert abs(kwh) <= TOL
            charge = discharge = 0.0
        elif act == "charge":
            charge, discharge = kwh, 0.0
            assert kwh <= battery["max_charge_kwh_per_hour"] + TOL
            assert h not in no_c
        else:
            charge, discharge = 0.0, kwh
            assert kwh <= battery["max_discharge_kwh_per_hour"] + TOL
            assert h not in no_d

        balance = g + s + discharge - hours_in[h]["demand_kwh"] - charge
        assert abs(balance) <= 0.05

        expected_after = prev_e + charge - discharge
        assert abs(expected_after - e_after) <= TOL
        assert e_after >= min_e[h] - TOL
        assert e_after <= battery["capacity_kwh"] + TOL
        if maxg[h] is not None:
            assert g <= maxg[h] + TOL
        prev_e = e_after

    assert abs(prev_e - battery["initial_energy_kwh"]) <= TOL


def compute_totals(hours_in, plan):
    total_g = sum(p["grid_kwh"] for p in plan)
    total_cost = sum(p["grid_kwh"] * hours_in[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
    peak = max(p["grid_kwh"] for p in plan)
    return round(total_g, 4), round(total_cost, 4), round(peak, 4)


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="GridWise LLM-Assisted Optimizer", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: ScenarioRequest):
    t0 = time.time()
    hours_dicts = [h.model_dump() for h in req.hours]
    battery = req.battery.model_dump()

    # 1. LLM
    try:
        raw = interpret_notes(req.operator_notes, battery)
    except Exception:
        log.exception("LLM stage failed")
        raise HTTPException(500, "LLM interpretation failed")

    # 2. Guardrails
    try:
        interp = validate_and_normalize(raw, len(req.operator_notes))
    except Exception as e:
        log.warning("Guardrail rejected LLM output: %s", e)
        interp = [
            {"note_index": i, "applies": False, "directive_type": "no_op",
             "structured_adjustment": None,
             "explanation": "Guardrail fallback."}
            for i in range(len(req.operator_notes))
        ]

    # 3. Optimize
    try:
        plan = solve_plan(hours_dicts, battery, interp)
    except Exception:
        log.exception("LP failed; retrying with all no_op")
        try:
            interp = [
                {"note_index": i, "applies": False, "directive_type": "no_op",
                 "structured_adjustment": None, "explanation": "solver fallback"}
                for i in range(len(req.operator_notes))
            ]
            plan = solve_plan(hours_dicts, battery, interp)
        except Exception:
            log.exception("LP fallback failed")
            raise HTTPException(500, "Optimization infeasible")

    # 4. Final validation
    try:
        validate_plan(hours_dicts, battery, interp, plan)
    except Exception:
        log.exception("Final validation failed")
        raise HTTPException(500, "Plan failed validation")

    total_g, total_cost, peak = compute_totals(hours_dicts, plan)

    applied = [e for e in interp if e["applies"]]
    if not applied:
        summary = "No applicable directives; schedule minimized under normal GridWise rules."
    else:
        kinds = ", ".join(sorted({e["directive_type"] for e in applied}))
        summary = f"Applied {len(applied)} directive(s): {kinds}. Optimized grid cost."

    log.info("optimize-energy done in %.3fs; cost=%.2f", time.time() - t0, total_cost)

    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interp,
        hourly_plan=plan,
        total_grid_kwh=total_g,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak,
        plan_summary=summary,
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    log.error("Unhandled: %s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": "internal error"})


# ============================================================
# CLI TEST MODE â€” runs 3 built-in test cases
# ============================================================
def _cli_test():
    """Run built-in sample tests without starting the server."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    print("\n" + "=" * 60)
    print(" GridWise â€” CLI Test Mode")
    print("=" * 60)

    # Health
    r = client.get("/health")
    print(f"\n[GET /health] -> {r.status_code} {r.json()}")
    assert r.status_code == 200

    # Test cases
    test_cases = [
        {
            "name": "Solar reduction + distractor",
            "payload": {
                "scenario_id": "TEST-01",
                "operator_notes": [
                    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
                    "The sports office moved next month's registration deadline."
                ],
                "hours": [
                    {"hour": h, "demand_kwh": 100 + h * 5, "solar_kwh": max(0, 100 - abs(h - 12) * 20),
                     "tariff_bdt_per_kwh": 5 + (h % 10)}
                    for h in range(24)
                ],
                "battery": {
                    "capacity_kwh": 200, "initial_energy_kwh": 100,
                    "minimum_energy_kwh": 30,
                    "max_charge_kwh_per_hour": 50,
                    "max_discharge_kwh_per_hour": 50,
                },
            },
        },
        {
            "name": "No-charge window",
            "payload": {
                "scenario_id": "TEST-02",
                "operator_notes": [
                    "The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance."
                ],
                "hours": [
                    {"hour": h, "demand_kwh": 100 + h * 3, "solar_kwh": max(0, 80 - abs(h - 12) * 15),
                     "tariff_bdt_per_kwh": 6 + (h % 8)}
                    for h in range(24)
                ],
                "battery": {
                    "capacity_kwh": 180, "initial_energy_kwh": 80,
                    "minimum_energy_kwh": 30,
                    "max_charge_kwh_per_hour": 50,
                    "max_discharge_kwh_per_hour": 50,
                },
            },
        },
        {
            "name": "Battery reserve (percent)",
            "payload": {
                "scenario_id": "TEST-03",
                "operator_notes": [
                    "Keep at least 50% of the battery capacity stored from 6 PM until 9 PM for emergency operations."
                ],
                "hours": [
                    {"hour": h, "demand_kwh": 110 + h * 4, "solar_kwh": max(0, 120 - abs(h - 12) * 25),
                     "tariff_bdt_per_kwh": 7 + (h % 9)}
                    for h in range(24)
                ],
                "battery": {
                    "capacity_kwh": 200, "initial_energy_kwh": 120,
                    "minimum_energy_kwh": 40,
                    "max_charge_kwh_per_hour": 50,
                    "max_discharge_kwh_per_hour": 50,
                },
            },
        },
    ]

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        r = client.post("/optimize-energy", json=tc["payload"])
        if r.status_code != 200:
            print(f"  [FAIL] status={r.status_code} body={r.text[:300]}")
            continue
        out = r.json()
        print(f"  Interp:")
        for e in out["directive_interpretation"]:
            print(f"    [{e['note_index']}] {e['directive_type']:25s} applies={e['applies']} adj={e['structured_adjustment']}")
        print(f"  Total grid:  {out['total_grid_kwh']:.2f} kWh")
        print(f"  Total cost:  {out['total_cost_bdt']:.2f} BDT")
        print(f"  Peak grid:   {out['peak_grid_kwh']:.2f} kWh")

    print("\n" + "=" * 60)
    print(" All tests completed.")
    print("=" * 60)


# ============================================================
# MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    import sys
    import uvicorn

    if "--test" in sys.argv:
        _cli_test()
    else:
        print("=" * 60)
        print(" GridWise LLM-Assisted Energy Optimizer")
        print("=" * 60)
        print(" Starting server on http://0.0.0.0:8000")
        print(" Interactive docs: http://localhost:8000/docs")
        print(" Health:           http://localhost:8000/health")
        print(" Run CLI tests:    python gridwise.py --test")
        print("=" * 60)
        uvicorn.run(app, host="0.0.0.0", port=8000)