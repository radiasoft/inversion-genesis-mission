"""lume_fbpic conversion of fbpic's `parametric_script.py` upstream example
(fbpic/docs/source/example_input/parametric_script.py): the `lwfa_script.py` physics (bare
pre-ionized electrons, linear density ramp, Gaussian laser, non-boosted) run once per value of
the laser amplitude `a0`, `[2.0, 4.0]`.

The simulation is identical to `lwfa.py`'s, so this script reuses its `build_model()`
and only adds the scan: each `a0` gets its own model and working directory, and
`model.set({"laser_a0": a0})` applies the value and runs the simulation -- the LUME way to
express a parametric scan, in place of upstream's MPI-rank bookkeeping.

Run from this directory or the `lume-fbpic` root; results land in `diags_a0_<a0>/diags/hdf5`.

Notable substitutions/approximations made converting it (not a bit-exact reproduction):

1. Parallelism: NOT reproduced, a real, acknowledged difference. Upstream launches
   `mpirun -np 2`, one `a0` per MPI rank, using `Simulation(..., use_all_mpi_ranks=False)` so
   the two ranks run independent simulations. `Simulation.setup_simulation()` hard-wires
   `use_all_mpi_ranks=self.hyparams.use_mpi`, so that mode is not reachable through
   `inversion_fbpic`. The scan below therefore runs the values one after another in one
   process. The physics of each run is unaffected; only the wall-clock time differs.

2. Everything else is `lwfa.py`'s conversion unchanged (see its module docstring):
   the exact linear-ramp `dens_func` (`LinearRampFlattop`), the `L_interact`/`v_window`
   settings that give upstream's exact step count, the exact inverse laser-duration
   conversion, and bare electrons via `species=None`. The one input that differs from
   `lwfa_script.py` is `p_zmin = 25e-6` instead of `30e-6`; it has no effect, because the
   relative density is 0 below `ramp_start = 30e-6` either way.

3. `p_rmax = 18e-6`: NOT reproduced, same as in `lwfa.py`. `_DensityProfile` does not
   pass `p_rmax` to `add_new_species()` (the field is only used for plot extents), so
   macroparticles are loaded out to `rmax = 20e-6` rather than 18e-6. `lwfa_script.py` has the
   same `p_rmax`, and `lwfa.py` still agreed with a real upstream run to within 2%.
"""

from __future__ import annotations

from pathlib import Path

from lwfa import build_model

A0_LIST = [2.0, 4.0]


def run_scan(a0_list: list[float] = A0_LIST) -> dict[float, dict[str, float]]:
    """Run one simulation per `a0` and return `{a0: {charge_pc, energy_mean_mev}}`."""
    results = {}
    for a0 in a0_list:
        model = build_model(working_directory=Path(f"diags_a0_{a0:.2f}"))
        model.simulator.configure()
        model.set({"laser_a0": a0})  # applies the value, then runs the simulation
        results[a0] = model.get(["charge_pc", "energy_mean_mev"])
    return results


if __name__ == "__main__":
    for a0, outputs in run_scan().items():
        print(f"a0 = {a0:.2f}: {outputs}")
