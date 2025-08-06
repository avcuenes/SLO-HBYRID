# SLO-HBYRID: Spiral + L-SHADE Hybrid with Eigen-DE and Stall Rescues

**SLO-HBYRID** is a bound-constrained black-box optimizer that fuses:
- **L-SHADE** core (success-history adaptation, linear pop-size reduction, archive)
- **Eigen-DE** crossover (periodic / on-stall, rotation-invariant exploitation)
- **Isotropic spiral** drift on the incumbent best (disabled near basin)
- **Rescues**: rand/1 fallback, Nelder–Mead polisher, in-place restarts
- **Early stop** with target & stall/diversity guards

> Code: Apache-2.0 • Docs & figures: CC-BY-4.0

## Quickstart
```bash
pip install numpy scipy  # scipy optional but recommended
