#!/bin/bash

# Runs one flavor of the test suite. Coverage is written to a data file, not
# xml: each CI cell keeps its own file and the coverage_gate job combines
# them, because no single OS covers every line (see runtests.yml).

# sysmon is coverage's sys.monitoring core, ~1.11x the no-coverage runtime
# against ~1.67x for the C tracer. It needs python >= 3.12 (the floor) and
# does not support branch coverage, which is off here.
export COVERAGE_CORE=sysmon

# -n logical rather than -n auto: xdist's auto asks psutil for *physical*
# cores, which is 2 on the SMT-enabled runners; logical gives all 3-4.
parallel=(-n logical --dist loadfile)
cov_args=(--cov dascore --cov-append --cov-report=)

# The doc tooling keeps its tests next to the scripts they test rather than
# under tests/, which mirrors the package. Named here so they are collected:
# without this the only thing running them is someone remembering to.
suite=(tests scripts docs/filters)

args=("${suite[@]}" -m "not network" "${parallel[@]}" "${cov_args[@]}")
if [[ "$1" == "network" ]]; then
  args=(tests -m network "${parallel[@]}" "${cov_args[@]}")
fi
if [[ "$1" == "doctest" ]]; then
  args=(dascore --doctest-modules)
fi
if [[ "$1" == "profile" ]]; then
  # No xdist: codspeed measures this process.
  args=(benchmarks --codspeed)
fi

python -c "
import os
try:
    import psutil
    physical = psutil.cpu_count(logical=False)
except ImportError:
    physical = None
print(f'cpus: logical={os.cpu_count()} physical={physical}')
"

python -m pytest "${args[@]}"
