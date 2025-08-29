from __future__ import annotations
import time, threading, psutil, hashlib
from contextlib import contextmanager
import pandas as pd

class MemSampler:
    def __init__(self, interval: float = 0.05):
        self.interval=interval; self._p=psutil.Process(); self._peak=0; self._run=False; self._th=None
    def start(self):
        self._run=True; self._peak=self._p.memory_info().rss
        self._th=threading.Thread(target=self._loop, daemon=True); self._th.start()
    def _loop(self):
        while self._run:
            rss=self._p.memory_info().rss
            if rss>self._peak: self._peak=rss
            time.sleep(self.interval)
    def stop(self)->int:
        self._run=False
        if self._th: self._th.join(timeout=1.0)
        return self._peak

@contextmanager
def timed_block():
    t0=time.perf_counter(); yield lambda: time.perf_counter()-t0

def pandas_signature(df: pd.DataFrame)->str:
    try:
        lim=min(2000, len(df)); sdf=df.copy()
        sdf=sdf.reindex(sorted(sdf.columns), axis=1).head(lim).sort_values(by=list(sdf.columns)).reset_index(drop=True)
        b=sdf.to_csv(index=False).encode("utf-8","ignore")
    except Exception:
        b=(str(df.shape)+str(list(df.dtypes.astype(str)))).encode()
    return hashlib.md5(b).hexdigest()
