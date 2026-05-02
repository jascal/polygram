# Polygram examples

## animals_interference.py

The canonical Polygram v0 walking tour. Declares a 4-feature, 2-cluster
Animals dictionary and sweeps `bird_hawk.phi` from 0 to π over 40 points,
watching the cross-cluster `(dog_poodle, bird_hawk)` overlap.

Run from the repo root:

```bash
python examples/animals_interference.py
```

Outputs land in `examples/output/`:
- `AnimalsInterference.q.orca.md` — verifiable Q-Orca artifact (parses +
  verifies clean against `q_orca>=0.7.1`)
- `run_AnimalsInterference.py` — self-contained reproducible runner
- `AnimalsInterference_result.npz` — full Gram tensor, overlaps,
  Schmidt ranks, assertion arrays
- `AnimalsInterference_result.csv` — flat per-sweep-point table for
  spreadsheet/plotting tools

`examples/output/` is gitignored.
