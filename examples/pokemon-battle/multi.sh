#!/bin/bash
set -euxo pipefail
parallel -N0 -j0 -u node build/index.js ::: {1..100}
