"""Verify frozen files and completeness without executing the Monte Carlo simulation."""
import csv, json, hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
P=ROOT/"results"/"final_200"
p=json.loads((P/"protocol.json").read_text())
for rel,expected in p["files_sha256"].items():
    actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    assert actual==expected, f"Hash mismatch: {rel}"
with (P/"episodes.csv").open(newline="",encoding="utf-8") as file:
    rows=list(csv.DictReader(file))
expected=p["n_seeds"]*p["episodes"]*len(p["scenarios"])*len(p["methods"])
assert len(rows)==expected, (len(rows),expected)
keys={(r["scenario"],r["method"],int(r["seed"]),int(r["episode"])) for r in rows}
assert len(keys)==expected,"Duplicate episode records"
assert sorted({int(r["seed"]) for r in rows})==list(range(p["seed_start"],p["seed_start"]+p["n_seeds"]))
counts=Counter(r["method"] for r in rows)
viol=sum(r["method"]=="Guarded" and float(r["energy_mj_step"])>p["energy_cap_mj_step"]+1e-12 for r in rows)
print("PASS: file hashes, unique/completed episode records, seed coverage")
print("Methods:",dict(counts))
print("Guarded episode violations:",viol)
