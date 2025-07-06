import duckdb
import time

DATA_PATH = "../data/train.csv"

con = duckdb.connect()

start = time.time()
df = con.execute(f"SELECT * FROM '{DATA_PATH}'").df()
duration = time.time() - start
print(f"[DuckDB] Loaded {df.shape[0]} rows in {duration:.3f} seconds")

# Example simple aggregation
start = time.time()
agg = con.execute(f"SELECT COUNT(*), AVG(Purchase) FROM '{DATA_PATH}'").fetchall()
duration = time.time() - start
print(f"[DuckDB] Aggregation query in {duration:.3f} seconds")