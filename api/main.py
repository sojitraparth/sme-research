import uuid, time, importlib, inspect
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd, duckdb
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE / "data" / "uploads"
RESULTS_CSV = BASE / "results" / "benchmarks.csv"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class RunRequest(BaseModel):
    dataset_id: Optional[str] = None
    engines: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    reps: int = 5
    cache_state: str = "warm"
    params: Dict[str, Any] = Field(default_factory=dict)

@app.get("/")
def root():
    return {"message": "sme-api ok"}

@app.get("/versions")
def versions():
    try:
        import polars as pl
        polars_ver = pl.__version__
    except Exception:
        polars_ver = None
    try:
        import pandas as pdm
        pandas_ver = pdm.__version__
    except Exception:
        pandas_ver = None
    return {"api": "1.0", "pandas": pandas_ver, "polars": polars_ver, "duckdb": duckdb.__version__}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".csv", ".parquet"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "Only CSV or Parquet supported"})
        did = uuid.uuid4().hex[:8]
        server_name = f"{did}_{Path(file.filename).name}"
        path = UPLOAD_DIR / server_name
        data = file.file.read()
        path.write_bytes(data)
        return {"ok": True, "dataset_id": did, "path": str(path)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.get("/results")
def results(limit: int = 2000):
    try:
        if not RESULTS_CSV.exists():
            return {"ok": True, "rows": []}
        df = pd.read_csv(RESULTS_CSV)
        return {"ok": True, "rows": df.tail(limit).to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.post("/run-benchmark")
def run_benchmark(req: RunRequest):
    try:
        mod = importlib.import_module("labbench.runner")
        fn = getattr(mod, "run_suite", None)
        if fn is None:
            return JSONResponse(status_code=500, content={"ok": False, "error": "labbench.runner.run_suite missing"})
        allowed = set(inspect.signature(fn).parameters.keys())
        params = dict(req.params or {})
        if req.dataset_id and "dataset_path" not in params:
            for p in UPLOAD_DIR.glob(f"{req.dataset_id}_*"):
                params["dataset_path"] = str(p)
                break
        candidate = {
            "engines": req.engines,
            "tasks": req.tasks,
            "reps": req.reps,
            "cache_state": req.cache_state,
            "params": params,
            "dataset_id": req.dataset_id,
            "uploads_dir": str(UPLOAD_DIR),
            "results_csv": str(RESULTS_CSV),
        }
        kwargs = {k: v for k, v in candidate.items() if k in allowed}
        t0 = time.time()
        run_id = fn(**kwargs)
        return {"ok": True, "run_id": run_id, "latency_s": time.time() - t0}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})