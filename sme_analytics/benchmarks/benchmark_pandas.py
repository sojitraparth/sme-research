import pandas as pd
import time

DATA_PATH = "../data/train.csv"

start = time.time()
df = pd.read_csv(DATA_PATH)
duration = time.time() - start
print(f"[Pandas] Loaded {df.shape[0]} rows in {duration:.3f} seconds")

# Example simple aggregation
start = time.time()
agg = df.describe()
duration = time.time() - start
print(f"[Pandas] Descriptive stats computed in {duration:.3f} seconds")