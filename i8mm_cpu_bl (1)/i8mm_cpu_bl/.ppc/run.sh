#!/usr/bin/env bash
set -e
cat > /box/i8mm.cc
chmod a-w /box/i8mm.cc

cd /program
/program/.ppc/grader.py --file "/box/i8mm.cc" --binary "/box/i8mm" --json "$@"
