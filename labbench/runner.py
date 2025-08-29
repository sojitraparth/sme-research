from __future__ import annotations
import os, gc, time, uuid, psutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

PROC = psutil.Process()

def _rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)

def _measure(fn):
    m0 = _rss_mb()
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    m1 = _rss_mb()
    return out, (t1 - t0), (m1 - m0)

def _load_pandas(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)

def _load_polars(path: str, dtype_mode: str = "numeric"):
    import polars as pl

    if path.lower().endswith(".csv"):
        try:
            df = pl.read_csv(
                path,
                infer_schema_length=10000,
                try_parse_dates=False,
                low_memory=False,
                ignore_errors=False,
            )
        except Exception:
            df = pl.read_csv(
                path,
                infer_schema_length=0,
                try_parse_dates=False,
                low_memory=False,
                ignore_errors=True,     
            )

        sample = df.head(min(20000, df.height))
        numeric_like = []
        for c in df.columns:
            try:
                s = sample[c].cast(pl.Utf8, strict=False).str.replace_all(",", "").str.replace_all("%", "")
                ok = s.cast(pl.Float64, strict=False).is_not_null().mean()
                if ok is not None and ok >= 0.8:
                    numeric_like.append(c)
            except Exception:
                pass

        if numeric_like:
            exprs = []
            for c in numeric_like:
                base = pl.col(c).cast(pl.Utf8, strict=False).str.replace_all(",", "")
                exprs.append(
                    pl.when(pl.col(c).cast(pl.Utf8, strict=False).str.contains(r"%$"))
                    .then(base.str.replace_all("%", "").cast(pl.Float64, strict=False) / 100.0)
                    .otherwise(base.cast(pl.Float64, strict=False))
                    .alias(c)
                )
            df = df.with_columns(exprs)

        if dtype_mode == "stringified":
            txt_cols = [c for c in df.columns if c not in numeric_like]
            if txt_cols:
                df = df.with_columns([pl.col(txt_cols).cast(pl.Utf8, strict=False)])
        return df

    df = pl.read_parquet(path)
    if dtype_mode == "stringified":
        num_cols = [c for c, dt in zip(df.columns, df.dtypes) if "Int" in str(dt) or "Float" in str(dt)]
        if num_cols:
            df = df.with_columns([pl.col(num_cols).cast(pl.Utf8, strict=False)])
    return df

def _load_duckdb(path: str):
    import duckdb
    df = _load_pandas(path)
    con = duckdb.connect()
    con.register("rel", df)
    rel = con.sql("SELECT * FROM rel")
    return con, rel

def _dataset_path_from_id(dataset_id: Optional[str], uploads_dir: str) -> Optional[str]:
    if not dataset_id:
        return None
    for p in Path(uploads_dir).glob(f"{dataset_id}_*"):
        return str(p)
    return None

def _sample_for_meta(path: str, n: int = 10000) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, nrows=n)
    df = pd.read_parquet(path)
    return df.head(n)

def _detect_cols(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    num_candidates = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_candidates.append(c)
            continue
        try:
            sn = pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
            if sn.notna().mean() >= 0.8:
                num_candidates.append(c)
        except Exception:
            pass
    cats = []
    for c in df.columns:
        u = df[c].nunique(dropna=True)
        if 2 <= u <= 1000 and not (c in num_candidates and pd.api.types.is_numeric_dtype(df[c])):
            cats.append(c)
    return num_candidates, cats

def _len_any(x: Any) -> Optional[int]:
    try:
        import polars as pl
        if hasattr(x, "shape"):
            return int(x.shape[0])
        if isinstance(x, pl.DataFrame):
            return int(x.height)
        if hasattr(x, "__len__"):
            return len(x)
    except Exception:
        pass
    try:
        return len(x)
    except Exception:
        return None

def run_suite(
    engines: List[str],
    tasks: List[str],
    reps: int,
    cache_state: str,
    params: Dict,
    dataset_id: Optional[str] = None,
    uploads_dir: str = "data/uploads",
    results_csv: str = "results/benchmarks.csv",
):
    run_id = uuid.uuid4().hex[:8]
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)

    dataset_path = _dataset_path_from_id(dataset_id, uploads_dir) or params.get("left_path") or params.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path not found")

    alias = {"filter_quantile": "filter", "groupby_agg": "groupby", "sort_values": "sort"}
    tasks = [alias.get(t, t) for t in tasks]

    dtype_mode = params.get("dtype_mode", "numeric")

    q_param = params.get("q", None)
    if q_param is None:
        selectivity_q = float(params.get("selectivity_q", 0.9))
    else:
        selectivity_q = float(q_param if float(q_param) <= 1 else float(q_param) / 100.0)

    sample = _sample_for_meta(dataset_path)
    num_cols, cat_cols = _detect_cols(sample)

    filter_col = params.get("filter_col")
    if not filter_col or filter_col not in sample.columns:
        filter_col = num_cols[0] if num_cols else None

    by = params.get("by")
    if isinstance(by, list) and by:
        by = by[0]
    if not by or by not in sample.columns:
        by = cat_cols[0] if cat_cols else (sample.columns[0] if len(sample.columns) else None)

    agg = params.get("agg", "sum")
    agg_col = params.get("agg_col")
    if (not agg_col or agg_col not in sample.columns) and num_cols:
        agg_col = num_cols[0] if num_cols[0] != by else (num_cols[1] if len(num_cols) > 1 else None)

    sort_col = params.get("sort_col")
    if not sort_col or sort_col not in sample.columns:
        sort_col = (num_cols[0] if num_cols else (sample.columns[0] if len(sample.columns) else None))

    topk = int(params.get("topk", 0))

    rows_out: list[dict] = []

    for engine in engines:
        def do_ingest():
            if engine == "pandas":
                return _load_pandas(dataset_path)
            if engine == "polars":
                return _load_polars(dataset_path, dtype_mode=dtype_mode)
            if engine == "duckdb":
                return _load_duckdb(dataset_path)
            raise ValueError(engine)

        df_obj = None

        if "ingest" in tasks:
            for r in range(reps):
                gc.collect()
                out, t, dm = _measure(do_ingest)
                nrows = None
                if engine == "pandas":
                    try: nrows = len(out)
                    except Exception: nrows = None
                elif engine == "polars":
                    try:
                        import polars as pl
                        nrows = out.height if isinstance(out, pl.DataFrame) else None
                    except Exception:
                        nrows = None
                else:
                    try: nrows = out[1].df().shape[0]  # rel -> df
                    except Exception: nrows = None
                rows_out.append({
                    "ts": time.time(), "run_id": run_id, "engine": engine, "task": "ingest",
                    "rep": r + 1, "time_s": t, "rss_mb": dm, "rows": nrows,
                    "cache_state": cache_state, "dataset": dataset_id
                })
                if cache_state == "warm" and df_obj is None:
                    df_obj = out
        else:
            if cache_state == "warm":
                df_obj = do_ingest()

        def get_obj():
            return df_obj if (cache_state == "warm" and df_obj is not None) else do_ingest()

        if "filter" in tasks:
            if not filter_col:
                rows_out.append({
                    "ts": time.time(), "run_id": run_id, "engine": engine, "task": "filter",
                    "rep": 0, "time_s": 0.0, "rss_mb": 0.0, "rows": None,
                    "cache_state": cache_state, "dataset": dataset_id
                })
            else:
                for r in range(reps):
                    gc.collect()
                    obj = get_obj()
                    try:
                        if engine == "pandas":
                            import pandas as pd
                            d = obj if isinstance(obj, pd.DataFrame) else (obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj))
                            s = pd.to_numeric(d[filter_col], errors="coerce")
                            thr = s.quantile(selectivity_q)
                            fn = (lambda: d[s > thr]) if True else (lambda: d[s < thr])
                            out, t, dm = _measure(fn)
                        elif engine == "polars":
                            import polars as pl
                            d = obj if hasattr(obj, "select") else pl.from_pandas(obj)
                            thr = d.select(pl.col(filter_col).quantile(selectivity_q)).item()
                            fn = lambda: d.filter(pl.col(filter_col) > thr)
                            out, t, dm = _measure(fn)
                        else:
                            con, rel = obj
                            sql = f"""
                                WITH q AS (SELECT quantile_cont(CAST({filter_col} AS DOUBLE), {selectivity_q}) AS th FROM rel)
                                SELECT * FROM rel, q WHERE CAST(rel.{filter_col} AS DOUBLE) > q.th
                            """
                            fn = lambda: con.execute(sql).df()
                            out, t, dm = _measure(fn)
                        nrows = _len_any(out)
                        rows_out.append({
                            "ts": time.time(), "run_id": run_id, "engine": engine, "task": "filter",
                            "rep": r + 1, "time_s": t, "rss_mb": dm, "rows": nrows,
                            "cache_state": cache_state, "dataset": dataset_id
                        })
                    except Exception:
                        rows_out.append({
                            "ts": time.time(), "run_id": run_id, "engine": engine, "task": "filter",
                            "rep": r + 1, "time_s": 0.0, "rss_mb": 0.0, "rows": None,
                            "cache_state": cache_state, "dataset": dataset_id
                        })

        if "groupby" in tasks:
            if not by:
                rows_out.append({
                    "ts": time.time(), "run_id": run_id, "engine": engine, "task": "groupby",
                    "rep": 0, "time_s": 0.0, "rss_mb": 0.0, "rows": None,
                    "cache_state": cache_state, "dataset": dataset_id
                })
            else:
                for r in range(reps):
                    gc.collect()
                    obj = get_obj()
                    try:
                        if engine == "pandas":
                            import pandas as pd
                            d = obj if isinstance(obj, pd.DataFrame) else (obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj))
                            if agg_col and agg in ("sum", "mean"):
                                fn = (lambda: d.groupby(by, observed=True)[agg_col].sum()) if agg == "sum" else (lambda: d.groupby(by, observed=True)[agg_col].mean())
                            elif agg_col:
                                fn = lambda: d.groupby(by, observed=True)[agg_col].count()
                            else:
                                fn = lambda: d.groupby(by, observed=True).size()
                            out, t, dm = _measure(fn)
                        elif engine == "polars":
                            import polars as pl
                            d = obj if hasattr(obj, "select") else pl.from_pandas(obj)
                            if agg_col and agg in ("sum", "mean"):
                                fn = (lambda: d.group_by(by).agg(pl.col(agg_col).sum())) if agg == "sum" else (lambda: d.group_by(by).agg(pl.col(agg_col).mean()))
                            elif agg_col:
                                fn = lambda: d.group_by(by).agg(pl.col(agg_col).count())
                            else:
                                fn = lambda: d.group_by(by).agg(pl.count())
                            out, t, dm = _measure(fn)
                        else:
                            con, rel = obj
                            if agg_col and agg in ("sum", "mean"):
                                metric = f"SUM(CAST({agg_col} AS DOUBLE))" if agg == "sum" else f"AVG(CAST({agg_col} AS DOUBLE))"
                            elif agg_col:
                                metric = f"COUNT({agg_col})"
                            else:
                                metric = "COUNT(*)"
                            sql = f"SELECT {by} AS g, {metric} AS metric FROM rel GROUP BY {by}"
                            fn = lambda: con.execute(sql).df()
                            out, t, dm = _measure(fn)
                        nrows = _len_any(out)
                        rows_out.append({
                            "ts": time.time(), "run_id": run_id, "engine": engine, "task": "groupby",
                            "rep": r + 1, "time_s": t, "rss_mb": dm, "rows": nrows,
                            "cache_state": cache_state, "dataset": dataset_id
                        })
                    except Exception:
                        rows_out.append({
                            "ts": time.time(), "run_id": run_id, "engine": engine, "task": "groupby",
                            "rep": r + 1, "time_s": 0.0, "rss_mb": 0.0, "rows": None,
                            "cache_state": cache_state, "dataset": dataset_id
                        })

        if "sort" in tasks:
            if not sort_col:
                rows_out.append({
                    "ts": time.time(), "run_id": run_id, "engine": engine, "task": "sort",
                    "rep": 0, "time_s": 0.0, "rss_mb": 0.0, "rows": None,
                    "cache_state": cache_state, "dataset": dataset_id
                })
            else:
                for r in range(reps):
                    gc.collect()
                    obj = get_obj()
                    try:
                        if engine == "pandas":
                            import pandas as pd
                            d = obj if isinstance(obj, pd.DataFrame) else (obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj))
                            if sort_col in d.columns:
                                if topk > 0 and pd.api.types.is_numeric_dtype(pd.to_numeric(d[sort_col], errors="coerce")):
                                    fn = lambda: d.nlargest(topk, sort_col)
                                else:
                                    fn = lambda: d.sort_values(sort_col, ascending=False)
                            else:
                                fn = lambda: d
                            out, t, dm = _measure(fn)
                        elif engine == "polars":
                            import polars as pl
                            d = obj if hasattr(obj, "select") else pl.from_pandas(obj)
                            if topk > 0:
                                fn = lambda: d.top_k(topk, by=sort_col)
                            else:
                                fn = lambda: d.sort(sort_col, descending=True)
                            out, t, dm = _measure(fn)
                        else:
                            con, rel = obj
                            sql = f"SELECT * FROM rel ORDER BY {sort_col} DESC"
                            if topk > 0:
                                sql += f" LIMIT {topk}"
                            fn = lambda: con.execute(sql).df()
                            out, t, dm = _measure(fn)
                        nrows = _len_any(out)
                        rows_out.append({
                            "ts": time.time(), "run_id": run_id, "engine": engine, "task": "sort",
                            "rep": r + 1, "time_s": t, "rss_mb": dm, "rows": nrows,
                            "cache_state": cache_state, "dataset": dataset_id
                        })
                    except Exception:
                        rows_out.append({
                            "ts": time.time(), "run_id": run_id, "engine": engine, "task": "sort",
                            "rep": r + 1, "time_s": 0.0, "rss_mb": 0.0, "rows": None,
                            "cache_state": cache_state, "dataset": dataset_id
                        })

        if isinstance(df_obj, tuple) and engine == "duckdb":
            try:
                df_obj[0].close()
            except Exception:
                pass

    out_df = pd.DataFrame(rows_out)
    if out_df.empty:
        raise RuntimeError("No benchmark rows produced")
    header = not os.path.exists(results_csv)
    out_df.to_csv(results_csv, mode="a", header=header, index=False)
    return run_id