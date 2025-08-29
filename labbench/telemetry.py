
# labbench/telemetry.py (NEW)
from __future__ import annotations
import threading, time, csv
from pathlib import Path
from typing import Optional
import psutil  # pip install psutil

class TelemetrySampler:
    """
    Samples process CPU% and RSS while a benchmark rep runs,
    and appends rows to results/telemetry.csv.
    """
    def __init__(self, run_id: str, engine: str, task: str, rep: int,
                 out_csv: Path, interval_s: float = 0.05):
        self.run_id, self.engine, self.task, self.rep = run_id, engine, task, rep
        self.out_csv = Path(out_csv)
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def _loop(self):
        t0 = time.perf_counter()
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        hdr = ["run_id","engine","task","rep","t_rel_s","rss_mb","cpu_pct"]
        new_file = not self.out_csv.exists()
        with self.out_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file: w.writerow(hdr)
            while not self._stop.is_set():
                t_rel = time.perf_counter() - t0
                mi = self._proc.memory_info()
                rss_mb = mi.rss / (1024*1024)
                # cpu_percent(None) uses delta from prior call; block on small interval for smoother series
                cpu = self._proc.cpu_percent(interval=None)
                w.writerow([self.run_id, self.engine, self.task, self.rep, f"{t_rel:.4f}", f"{rss_mb:.2f}", f"{cpu:.1f}"])
                time.sleep(self.interval_s)

    def __enter__(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thr: self._thr.join()