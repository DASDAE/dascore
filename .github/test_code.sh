#!/bin/bash

# Script to run tests to account for wonkiness of periodic mac failures.
args="tests -s --cov dascore --cov-append --cov-report=xml"
if [[ "$1" == "doctest" ]]; then
  args="dascore --doctest-modules"
fi
if [[ "$1" == "benchmarks" ]]; then
  args="benchmarks --codspeed"
fi

exit_code=0

python -m pytest $args || exit_code=$?

# Check the exit code is related to sporadic failures on mac, see #312
if [ $exit_code -ne 132 ] && [ $exit_code -ne 0 ]; then
  exit $exit_code
fi
