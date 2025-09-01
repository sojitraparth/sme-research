# app.py
import os, io, time, json, requests, pandas as pd, numpy as np, psutil
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────── PAGE CONFIG ──────────────────────────────
st.set_page_config(page_title="SME Research", layout="wide", initial_sidebar_state="expanded")

# Force a Light-mode visual layer by default (no auto-dark).
# Note: Streamlit's core theme can't be changed programmatically; this CSS ensures
# a light look regardless of user/system auto-dark, and you can opt-in to Dark in Settings.
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

def _apply_theme():
    theme = st.session_state.get("theme", "Light")
    if theme == "Light":
        st.markdown("""
        <style>
        :root { --bg:#f7f7fb; --panel:#ffffff; --border:#ececf2; --text:#111827; }
        body { background: var(--bg); }
        section[data-testid="stSidebar"] { background: var(--panel); border-right:1px solid var(--border); }
        div[data-testid='stMetricValue'] { font-weight:700; }
        .card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; margin-bottom:10px; }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Simple Dark variant (opt-in only)
        st.markdown("""
        <style>
        :root { --bg:#0f1220; --panel:#151a2d; --border:#2b3150; --text:#e5e7eb; }
        body { background: var(--bg); }
        section[data-testid="stSidebar"] { background: var(--panel); border-right:1px solid var(--border); }
        div[data-testid='stMetricValue'] { font-weight:700; color:var(--text); }
        .card { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; margin-bottom:10px; color:var(--text); }
        .stMarkdown, .stText, .stDataFrame, .stPlotlyChart, .stCaption, .stMetric { color:var(--text) !important; }
        </style>
        """, unsafe_allow_html=True)

_apply_theme()

# ────────────────────────────── CONSTANTS ──────────────────────────────
BASE_DIR = Path(__file__).parent
API = os.environ.get("SME_API", "http://localhost:8000")
RESULTS_CSV = Path("results/benchmarks.csv")
LATENCY_CSV = Path("results/latency.csv")
SWITCH_CSV = Path("results/switch_local_metrics.csv")
_process = psutil.Process()

# ────────────────────────────── CACHED RESOURCES ──────────────────────────────
@st.cache_resource
def duck_con():
    import duckdb
    return duckdb.connect()

# ────────────────────────────── UTILITIES / API ──────────────────────────────
def _rss_mb():
    return _process.memory_info().rss / (1024 * 1024)

def api_versions():
    try:
        r = requests.get(f"{API}/versions", timeout=5)
        return r.json() if r.ok else {"error": f"{r.status_code} {r.text}"}
    except Exception as e:
        return {"error": str(e)}

def api_upload(filedict):
    try:
        r = requests.post(
            f"{API}/upload",
            files={"file": (filedict["name"], filedict["bytes"], filedict.get("mime", "application/octet-stream"))},
            timeout=120,
        )
        return r.json() if r.ok else {"error": f"{r.status_code} {r.text}"}
    except Exception as e:
        return {"error": str(e)}

def api_run(payload):
    try:
        r = requests.post(f"{API}/run-benchmark", json=payload, timeout=3600)
        return r.json() if r.ok else {"error": f"{r.status_code} {r.text}"}
    except Exception as e:
        return {"error": str(e)}

def api_results():
    try:
        r = requests.get(f"{API}/results", timeout=5)
        return r.json() if r.ok else {"error": f"{r.status_code} {r.text}"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def load_df_from_bytes(name, blob):
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(blob))
    if name.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(blob))
    raise ValueError("Unsupported file type. Use CSV or Parquet.")

def read_results_csv():
    if RESULTS_CSV.exists():
        try:
            return pd.read_csv(RESULTS_CSV)
        except Exception:
            return None
    return None

def pretty_seconds(x):
    try:
        return f"{float(x):.3f}s"
    except Exception:
        return str(x)

def pick_numeric(df):
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

def parse_dates_safe(series, fmt):
    if fmt.strip():
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def coerce_numeric_candidates(df):
    nums = pick_numeric(df)
    if nums:
        return nums
    cand = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(len(df) * 0.05)):
            cand.append(c)
    return cand

def candidate_categories(df, max_card=200):
    cats = []
    for c in df.columns:
        try:
            u = df[c].nunique(dropna=True)
            if 2 <= u <= max_card:
                cats.append(c)
        except Exception:
            pass
    return cats

def _append_switch_log(row):
    SWITCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SWITCH_CSV.exists()
    pd.DataFrame([row]).to_csv(SWITCH_CSV, mode="a", header=write_header, index=False)

def _auto_cols_for_local(df):
    nums = coerce_numeric_candidates(df)
    agg_col = nums[0] if nums else None
    cats = candidate_categories(df)
    by = cats[0] if cats else None
    sort_col = agg_col
    filter_col = agg_col
    return df, by, agg_col, sort_col, filter_col

def measure_local_engine(df, engine, sample_cap=200_000, q=0.9, topk=1000):
    if df is None or len(df) == 0:
        return {"ok": False, "reason": "empty_df"}
    if len(df) > sample_cap:
        df = df.sample(sample_cap, random_state=42).reset_index(drop=True)
    df, by, agg_col, sort_col, filter_col = _auto_cols_for_local(df.copy())
    if agg_col is None:
        return {"ok": False, "reason": "no_numeric_cols"}

    t0 = time.perf_counter()
    m0 = _rss_mb()
    t_groupby = t_filter = t_sort = None
    rows_g = rows_f = rows_s = None

    try:
        if engine == "pandas":
            d = df
            if by is not None:
                t1 = time.perf_counter()
                g = d.groupby(by, observed=True)[agg_col].mean()
                t_groupby = time.perf_counter() - t1
                rows_g = int(len(g))
            if filter_col is not None:
                t1 = time.perf_counter()
                thr = d[filter_col].quantile(q)
                f = d[d[filter_col] > thr]
                t_filter = time.perf_counter() - t1
                rows_f = int(len(f))
            if sort_col is not None:
                t1 = time.perf_counter()
                s = d.nlargest(min(topk, len(d)), sort_col) if sort_col in d.columns else d.head(min(topk, len(d)))
                t_sort = time.perf_counter() - t1
                rows_s = int(len(s))

        elif engine == "polars":
            import polars as pl
            d = pl.from_pandas(df)
            if by is not None:
                t1 = time.perf_counter()
                g = d.group_by(by).agg(pl.col(agg_col).mean())
                t_groupby = time.perf_counter() - t1
                rows_g = int(g.height)
            if filter_col is not None:
                t1 = time.perf_counter()
                # get scalar via Series before .item()
                thr = d.select(pl.col(filter_col).quantile(q)).to_series().item()
                f = d.filter(pl.col(filter_col) > thr)
                t_filter = time.perf_counter() - t1
                rows_f = int(f.height)
            if sort_col is not None:
                t1 = time.perf_counter()
                s = d.top_k(min(topk, d.height), by=sort_col)  # descending
                t_sort = time.perf_counter() - t1
                rows_s = int(s.height)

        elif engine == "duckdb":
            con = duck_con()
            con.register("t", df)
            if by is not None:
                t1 = time.perf_counter()
                g = con.execute(f"SELECT {by} AS g, AVG({agg_col}) AS m FROM t GROUP BY {by}").df()
                t_groupby = time.perf_counter() - t1
                rows_g = int(len(g))
            if filter_col is not None:
                t1 = time.perf_counter()
                thr = con.execute(f"SELECT quantile_cont({filter_col}, {q}) FROM t WHERE {filter_col} IS NOT NULL").fetchone()[0]
                f = con.execute(f"SELECT * FROM t WHERE {filter_col} > {thr}").df()
                t_filter = time.perf_counter() - t1
                rows_f = int(len(f))
            if sort_col is not None:
                t1 = time.perf_counter()
                s = con.execute(f"SELECT * FROM t ORDER BY {sort_col} DESC LIMIT {min(topk, len(df))}").df()
                t_sort = time.perf_counter() - t1
                rows_s = int(len(s))
        else:
            return {"ok": False, "reason": f"unknown_engine:{engine}"}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    mem_delta = _rss_mb() - m0
    t_total = time.perf_counter() - t0
    return {
        "ok": True,
        "engine": engine,
        "rows_sampled": int(len(df)),
        "t_total_s": float(t_total),
        "mem_delta_mb": float(mem_delta),
        "t_groupby_s": float(t_groupby) if t_groupby is not None else None,
        "t_filter_s": float(t_filter) if t_filter is not None else None,
        "t_sort_s": float(t_sort) if t_sort is not None else None,
        "rows_g": rows_g, "rows_f": rows_f, "rows_s": rows_s,
    }

# ────────────────────────────── GLOBAL ENGINE SELECTOR ──────────────────────────────
def engine_selector():
    """Global engine selector with segmented-control fallback; updates session and probes."""
    engines = ["pandas", "polars", "duckdb"]
    current = st.session_state.get("ui_engine", "pandas")

    if hasattr(st, "segmented_control"):
        sel = st.segmented_control(
            "Engine", engines,
            default=current,
            key="ui_engine_seg",
            help="Select local processing engine for this session",
            width="content",
        )
    else:
        sel = st.radio(
            "Engine", engines,
            index=engines.index(current) if current in engines else 0,
            horizontal=True,
            key="ui_engine_radio",
            help="Select local processing engine for this session",
        )

    if sel != current:
        st.session_state["last_local_engine"] = current
        st.session_state["ui_engine"] = sel

        df_local = st.session_state.get("df_data")
        if df_local is not None:
            try:
                m = measure_local_engine(df_local, sel)
                if m.get("ok"):
                    st.session_state.setdefault("last_local_metrics", {})[sel] = m
                    tg, tf, ts_ = m.get("t_groupby_s"), m.get("t_filter_s"), m.get("t_sort_s")
                    def _fmt(v): return f"{v:.3f}s" if v is not None else "n/a"
                    st.info(
                        f"Switched to {sel}. Total {m['t_total_s']:.3f}s, ΔRSS {m['mem_delta_mb']:.1f} MB. "
                        f"Groupby {_fmt(tg)}, Filter {_fmt(tf)}, Sort {_fmt(ts_)}."
                    )
                else:
                    st.warning(f"Engine probe skipped: {m.get('reason','unknown')}")
            except Exception as e:
                st.warning(f"Probe error: {e}")

        st.rerun()  # refresh rest of the app

# ────────────────────────────── HEADER ──────────────────────────────
hdr_l, hdr_r = st.columns([0.75, 0.25])
with hdr_l:
    st.title("SME Research Lab")
    st.caption("Task-centric benchmarking and SME analytics with Pandas, Polars, and DuckDB")
with hdr_r:
    ver = api_versions()
    if "error" not in ver:
        st.success("API: Online", icon="✅")
    else:
        st.error("API: Offline", icon="🛑")
        st.caption(f"Details: {ver.get('error')}")
    engine_selector()  # ← global engine switcher in header

# ────────────────────────────── NAVIGATION ──────────────────────────────
nav = st.sidebar.radio(
    "Navigation",
    ["Upload & Clean", "SME Benchmarks", "Dashboard", "Visualize", "Forecast", "Summary", "Download", "Settings"],
    index=0, key="nav_main",
)

def get_df():
    if "segmented_df" in st.session_state:
        return st.session_state["segmented_df"]
    if "df_data" in st.session_state:
        return st.session_state["df_data"]
    return None

# ────────────────────────────── UPLOAD & CLEAN ──────────────────────────────
if nav == "Upload & Clean":
    st.subheader("Upload & Clean")
    c1, c2 = st.columns([0.55, 0.45])

    with c1:
        up = st.file_uploader("CSV or Parquet", type=["csv", "parquet"], key="uploader_main")
        if up is not None:
            st.session_state["upload_name"] = up.name
            st.session_state["upload_bytes"] = up.getvalue()
            st.session_state["upload_mime"] = getattr(up, "type", "application/octet-stream")

        has_persisted = "upload_bytes" in st.session_state and "upload_name" in st.session_state
        if has_persisted:
            try:
                df_prev = load_df_from_bytes(st.session_state["upload_name"], st.session_state["upload_bytes"])
                st.dataframe(df_prev.head(200), use_container_width=True)
                colA, colB, colC = st.columns(3)
                with colA:
                    if st.button("Load into session", key="btn_load_session"):
                        st.session_state["df_data"] = df_prev
                        st.success(f"Loaded {df_prev.shape[0]:,} rows × {df_prev.shape[1]:,} columns")
                with colB:
                    if st.button("Upload to benchmark API", key="btn_upload_api"):
                        meta = api_upload({"name": st.session_state["upload_name"], "bytes": st.session_state["upload_bytes"], "mime": st.session_state.get("upload_mime")})
                        if "error" in meta:
                            st.error(f"Upload failed: {meta['error']}")
                        else:
                            st.session_state["dataset_id"] = meta.get("dataset_id")
                            st.session_state["dataset_path"] = meta.get("path")
                            st.success(f"Uploaded. dataset_id={meta.get('dataset_id')}")
                with colC:
                    if st.button("Clear uploaded file", key="btn_clear_upload"):
                        for k in ["upload_name","upload_bytes","upload_mime"]:
                            st.session_state.pop(k, None)
                        st.rerun()
            except Exception as e:
                st.error(f"Preview/Load failed: {e}")
        else:
            st.info("Choose a file to preview or load")

    with c2:
        st.markdown("Segmentation Filters")
        if "df_data" in st.session_state:
            df_seg = st.session_state["df_data"].copy()
            for col in df_seg.select_dtypes(include=["object","category"]).columns:
                if df_seg[col].nunique() <= 50:
                    choices = sorted(map(str, df_seg[col].dropna().unique()))
                    sel = st.multiselect(col, choices, default=choices, key=f"seg_{col}")
                    if sel:
                        df_seg = df_seg[df_seg[col].astype(str).isin(sel)]
            st.session_state["segmented_df"] = df_seg
            st.success(f"Filters applied: {df_seg.shape[0]:,} rows")
            if st.button("Clear filters", key="btn_clear_filters"):
                st.session_state.pop("segmented_df", None)
                st.rerun()
        else:
            st.info("Load a dataset to enable filters")

        st.markdown("---")
        if st.button("Clear current dataset", key="btn_clear_dataset"):
            for k in ["df_data","segmented_df"]:
                st.session_state.pop(k, None)
            st.success("Cleared current dataset")

# ────────────────────────────── BENCHMARKS ──────────────────────────────
if nav == "SME Benchmarks":
    st.subheader("Configure & Run Benchmarks")
    if "dataset_id" not in st.session_state:
        st.info("Upload a dataset on the Upload & Clean page to enable API benchmarks")

    engines = st.multiselect("Engines", ["pandas", "polars", "duckdb"], default=["pandas", "polars", "duckdb"], key="bm_engines")

    df_here = get_df()
    num_cols = coerce_numeric_candidates(df_here) if df_here is not None else []
    cat_cols = candidate_categories(df_here) if df_here is not None else []

    task_options, task_defaults = [], []
    task_options.append("ingest"); task_defaults.append("ingest")
    if num_cols:
        task_options += ["filter", "sort"]; task_defaults += ["filter", "sort"]
    if cat_cols and num_cols:
        task_options.append("groupby"); task_defaults.append("groupby")

    tasks = st.multiselect("Tasks", task_options, default=task_defaults, key="bm_tasks")

    reps = st.number_input("Repetitions per task", min_value=3, max_value=50, value=7, step=1, key="bm_reps")
    cache_state = st.selectbox("Cache state", ["warm", "cold"], index=0, key="bm_cache")

    params = {}
    if "filter" in tasks:
        fcol = st.selectbox("Filter numeric column", options=num_cols if num_cols else ["value"], index=0, key="bm_fq_col")
        q = st.slider("Filter quantile", min_value=0.5, max_value=0.99, value=0.9, step=0.01, key="bm_fq_q")
        ftype = st.selectbox("Filter type", ["greater_than_quantile", "less_than_quantile"], index=0, key="bm_fq_type")
        params.update({"filter_col": fcol, "q": float(q), "filter_type": ftype})

    if "groupby" in tasks:
        by = st.selectbox("Group key", cat_cols if cat_cols else ["<none>"], index=0, key="bm_ga_by")
        agg_col = st.selectbox("Aggregate column", num_cols if num_cols else ["<none>"], index=0, key="bm_ga_col")
        agg = st.selectbox("Aggregation", ["sum", "mean", "count"], index=0, key="bm_ga_fn")
        if by != "<none>":
            params["by"] = by
        if agg_col != "<none>":
            params["agg_col"] = agg_col
        params["agg"] = agg
        tg = st.number_input("Target groups (hash buckets)", min_value=10, max_value=100000, value=1000, step=10, key="bm_ga_tg")
        params["target_groups"] = int(tg)

    if "sort" in tasks:
        sc = st.selectbox("Sort column", options=num_cols if num_cols else ["value"], index=0, key="bm_sort_col")
        asc = st.checkbox("Ascending", value=False, key="bm_sort_asc")
        tk = st.number_input("Top-K (0 for full sort)", min_value=0, max_value=1000000, value=1000, step=100, key="bm_sort_topk")
        params.update({"sort_col": sc, "ascending": bool(asc), "topk": int(tk)})

    can_run = "dataset_id" in st.session_state and len(tasks) > 0 and len(engines) > 0
    if st.button("Run benchmark", disabled=not can_run, key="bm_run_btn"):
        payload = {
            "dataset_id": st.session_state.get("dataset_id"),
            "engines": engines,
            "tasks": tasks,
            "reps": int(reps),
            "cache_state": cache_state,
            "params": params,
        }
        t0 = time.perf_counter()
        res = api_run(payload)
        t1 = time.perf_counter()
        if "error" in res:
            st.error(f"Run failed: {res['error']}")
            det = res.get("detail") or res.get("error")
            if isinstance(det, str):
                with st.expander("Details"):
                    st.code(det)
        else:
            LATENCY_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"ts": time.time(), "action": "benchmark_run", "lat_s": t1 - t0}]).to_csv(
                LATENCY_CSV, mode="a", header=not LATENCY_CSV.exists(), index=False
            )
            st.success(f"Run complete. id={res.get('run_id')}")

    st.markdown("Results summary")
    dfb = read_results_csv()
    if dfb is None or dfb.empty:
        st.info("No results yet")
    else:
        st.dataframe(dfb.tail(500), use_container_width=True)
        if all(x in dfb.columns for x in ["task", "engine"]):
            agg_cols = [c for c in ["time_s", "rss_mb"] if c in dfb.columns]
            if agg_cols:
                agg = dfb.groupby(["task", "engine"])[agg_cols].mean().reset_index()
                if "time_s" in agg.columns:
                    fig1 = px.bar(agg, x="engine", y="time_s", color="engine", facet_col="task", facet_col_wrap=3,
                                  title="Mean execution time (s) by engine per task", height=480)
                    st.plotly_chart(fig1, use_container_width=True)
                if "rss_mb" in agg.columns:
                    fig2 = px.bar(agg, x="engine", y="rss_mb", color="engine", facet_col="task", facet_col_wrap=3,
                                  title="Mean peak RSS (MB) by engine per task", height=480)
                    st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────── DASHBOARD ──────────────────────────────
if nav == "Dashboard":
    st.subheader("Dashboard")
    df = get_df()
    if df is None:
        st.info("Load data in Upload & Clean")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{len(df):,}")
        k2.metric("Columns", f"{df.shape[1]:,}")
        k3.metric("Numeric", f"{len(pick_numeric(df)):,}")
        k4.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum()/1e6:,.2f}")

        st.markdown("Missing data")
        miss = (df.isna().mean() * 100).loc[lambda s: s > 0].sort_values(ascending=False)
        if not miss.empty:
            figm = px.bar(miss.reset_index().rename(columns={"index": "column", 0: "pct_missing"}),
                          x="pct_missing", y="column", orientation="h",
                          title="Percent Missing by Column", height=420)
            st.plotly_chart(figm, use_container_width=True)

        st.markdown("Top and bottom records")
        nums = pick_numeric(df)
        if nums:
            metric = st.selectbox("Rank by", nums, index=0, key="dash_rank_metric")
            n = st.slider("Rows", 5, 50, 10, key="dash_rank_n")
            ranked = df.sort_values(metric, ascending=False)
            top = ranked.head(n).assign(Rank=range(1, n + 1))
            bot = ranked.tail(n).assign(Rank=range(1, n + 1))
            t1, t2 = st.tabs([f"Top {n}", f"Bottom {n}"])
            with t1:
                st.dataframe(top[["Rank", metric]], use_container_width=True)
            with t2:
                st.dataframe(bot[["Rank", metric]], use_container_width=True)
        else:
            st.info("No numeric columns found")

# ────────────────────────────── VISUALIZE ──────────────────────────────
if nav == "Visualize":
    st.subheader("Visualize")
    df = get_df()
    if df is None:
        st.info("Load data in Upload & Clean")
    else:
        nums = pick_numeric(df)
        if nums:
            st.dataframe(df[nums].describe().T, use_container_width=True)
            if len(nums) > 1:
                figc = px.imshow(df[nums].corr(), text_auto=True, title="Correlation matrix")
                st.plotly_chart(figc, use_container_width=True)
            for c in nums:
                st.plotly_chart(px.histogram(df, x=c, nbins=30, title=f"Histogram: {c}"), use_container_width=True)
                st.plotly_chart(px.box(df, y=c, title=f"Outliers: {c}"), use_container_width=True)
        cats = [c for c in df.select_dtypes(include=["object","category"]).columns]
        for c in cats[:6]:
            counts = df[c].value_counts().rename_axis(c).reset_index(name="count")
            fb = px.bar(counts, x=c, y="count", title=f"Distribution: {c}")
            st.plotly_chart(fb, use_container_width=True)

# ────────────────────────────── FORECAST ──────────────────────────────
if nav == "Forecast":
    st.subheader("Forecast")
    df = get_df()
    if df is None:
        st.info("Load data in Upload & Clean")
    else:
        date_cols = [c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().sum() >= 5]
        nums = pick_numeric(df)
        if not date_cols or not nums:
            st.warning("Need at least one date column and one numeric column")
        else:
            d_col = st.selectbox("Date column", date_cols, index=0, key="fc_date_col")
            v_col = st.selectbox("Value column", nums, index=0, key="fc_val_col")
            fmt = st.text_input("Date format (optional, e.g. %Y-%m-%d)", value="", key="fc_fmt")
            horizon = st.slider("Horizon (days)", 7, 365, 30, key="fc_horizon")
            split = st.slider("Train %", 50, 90, 80, key="fc_split")
            models = st.multiselect("Models", ["Prophet", "ARIMA", "Holt-Winters"], default=["Prophet"], key="fc_models")
            ds = parse_dates_safe(df[d_col], fmt)
            y = pd.to_numeric(df[v_col], errors="coerce")
            ts = pd.DataFrame({"ds": ds, "y": y}).dropna()
            ts.sort_values("ds", inplace=True)
            ts.reset_index(drop=True, inplace=True)
            if ts.empty:
                st.error("No valid rows after parsing")
            else:
                cut = int(len(ts) * split / 100)
                train = ts.iloc[:cut].copy()
                test = ts.iloc[cut:].copy()
                if st.button("Run forecast", key="fc_run"):
                    perf, fcs = [], {}
                    if "Prophet" in models:
                        try:
                            from prophet import Prophet
                            m = Prophet()
                            m.fit(train)
                            fut = m.make_future_dataframe(periods=horizon)
                            pr = m.predict(fut)[["ds", "yhat"]].set_index("ds").rename(columns={"yhat": "yhat_prophet"})
                            fcs["Prophet"] = pr
                            pred = m.predict(test[["ds"]])[["ds", "yhat"]].set_index("ds").rename(columns={"yhat": "yhat_prophet"})
                            dfj = test.set_index("ds")["y"].to_frame().join(pred, how="inner")
                            rmse = float(np.sqrt(((dfj.y - dfj.yhat_prophet) ** 2).mean()))
                            mape = float((np.abs(dfj.y - dfj.yhat_prophet) / dfj.y.replace(0, np.nan)).dropna().mean() * 100)
                            perf.append({"Model": "Prophet", "RMSE": rmse, "MAPE": mape})
                        except Exception as e:
                            st.error(f"Prophet error: {e}")
                    if "ARIMA" in models:
                        try:
                            import statsmodels.api as sm
                            p = st.number_input("ARIMA p", 0, 5, 1, key="fc_arima_p")
                            d = st.number_input("ARIMA d", 0, 2, 1, key="fc_arima_d")
                            q = st.number_input("ARIMA q", 0, 5, 1, key="fc_arima_q")
                            ar = sm.tsa.ARIMA(train["y"], order=(p, d, q)).fit()
                            ar_fc = ar.forecast(steps=horizon)
                            idx = pd.date_range(start=train.ds.iloc[-1], periods=horizon + 1, inclusive="right")
                            df_ar = pd.DataFrame({"yhat_arima": ar_fc.values}, index=idx)
                            fcs["ARIMA"] = df_ar
                            tst = test.set_index("ds")["y"].to_frame()
                            dfj2 = tst.join(df_ar.reindex(tst.index).dropna(), how="inner")
                            rmse = float(np.sqrt(((dfj2.y - dfj2.yhat_arima) ** 2).mean()))
                            mape = float((np.abs(dfj2.y - dfj2.yhat_arima) / dfj2.y.replace(0, np.nan)).dropna().mean() * 100)
                            perf.append({"Model": "ARIMA", "RMSE": rmse, "MAPE": mape})
                        except Exception as e:
                            st.warning(f"ARIMA skipped: {e}")
                    if "Holt-Winters" in models:
                        try:
                            from statsmodels.tsa.holtwinters import ExponentialSmoothing
                            hw = ExponentialSmoothing(train["y"]).fit()
                            hw_fc = hw.forecast(horizon)
                            idx2 = pd.date_range(start=train.ds.iloc[-1], periods=horizon + 1, inclusive="right")
                            df_hw = pd.DataFrame({"yhat_hw": hw_fc.values}, index=idx2)
                            fcs["Holt-Winters"] = df_hw
                            tst2 = test.set_index("ds")["y"].to_frame()
                            dfj3 = tst2.join(df_hw.reindex(tst2.index).dropna(), how="inner")
                            rmse = float(np.sqrt(((dfj3.y - dfj3.yhat_hw) ** 2).mean()))
                            mape = float((np.abs[dfj3.y - dfj3.yhat_hw] / dfj3.y.replace(0, np.nan)).dropna().mean() * 100) if len(dfj3) else float("nan")
                            perf.append({"Model": "Holt-Winters", "RMSE": rmse, "MAPE": mape})
                        except Exception as e:
                            st.warning(f"Holt-Winters skipped: {e}")
                    if perf:
                        st.markdown("Performance")
                        st.dataframe(pd.DataFrame(perf), use_container_width=True)
                        for name, df_fc in fcs.items():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=train.ds, y=train.y, name="Train"))
                            fig.add_trace(go.Scatter(x=test.ds, y=test.y, name="Test"))
                            fig.add_trace(go.Scatter(x=df_fc.index, y=df_fc.iloc[:, 0], name=name))
                            fig.update_layout(title=f"{name} Forecast vs Actual", xaxis_title="Date", yaxis_title=v_col)
                            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────── SUMMARY ──────────────────────────────
if nav == "Summary":
    st.subheader("Executive Summary")
    df = get_df()
    if df is None:
        st.info("Load data in Upload & Clean")
    else:
        numeric = pick_numeric(df)
        kpis = st.multiselect("KPIs", options=numeric, default=numeric[:4] if numeric else [], key="sum_kpis")
        if kpis:
            cols = st.columns(len(kpis))
            for i, c in enumerate(kpis):
                total = df[c].sum()
                mean = df[c].mean()
                med = df[c].median()
                cols[i].metric(label=c, value=f"{total:,.0f}", delta=f"μ={mean:.1f}, ˜={med:.1f}")
        st.divider()
        cats = [c for c in df.select_dtypes(include=["object","category"]).columns if df[c].nunique() <= 50]
        if cats:
            cat = st.selectbox("Category breakdown", cats, key="sum_cat")
            counts = df[cat].value_counts().rename_axis(cat).reset_index(name="count")
            fig = px.pie(counts, names=cat, values="count", title=f"{cat} Distribution")
            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────── DOWNLOADS ──────────────────────────────
if nav == "Download":
    st.subheader("Downloads")
    df = get_df()
    if df is None:
        st.info("Load data in Upload & Clean")
    else:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download current dataset (CSV)", data=csv, file_name="dataset_current.csv", mime="text/csv", key="dl_dataset")
    dfb = read_results_csv()
    if dfb is not None and not dfb.empty:
        csvb = dfb.to_csv(index=False).encode("utf-8")
        st.download_button("Download benchmarks.csv", data=csvb, file_name="benchmarks.csv", mime="text/csv", key="dl_bench")

# ────────────────────────────── SETTINGS ──────────────────────────────
if nav == "Settings":
    st.subheader("Settings")
    theme = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True, key="set_theme",
                     help="Light mode is the default; Dark is opt-in only.")
    st.session_state["theme"] = theme
    _apply_theme()

    export_format = st.selectbox("Default export format", ["CSV", "Excel (.xlsx)"], index=0, key="set_export_fmt")
    st.session_state["export_format"] = export_format
    default_date_fmt = st.text_input("Default Date Format", value=st.session_state.get("date_format", "%Y-%m-%d"), key="set_datefmt")
    st.session_state["date_format"] = default_date_fmt
    default_email = st.text_input("Notification Email", value=st.session_state.get("user_email", ""), key="set_email")
    st.session_state["user_email"] = default_email
    if st.button("Save", key="set_save"):
        st.success("Saved")