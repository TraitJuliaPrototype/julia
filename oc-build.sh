#!/bin/bash
# Cold build of the traits-v2 base (vanilla 1.13.2 @ 45b6d15e34), niced, -j4, memory watchdog
# (kill below 4 GB available), after the pattern of the old tree's tw-build.sh.
cd "$(dirname "$0")"
setsid nice -n 19 make -j4 ${OC_TARGET:-} > oc-build.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt 4096 ]; then echo "WATCHDOG: MemAvailable ${avail} MB, killing build" >> oc-build.log; kill -TERM -$pid; sleep 5; kill -KILL -$pid 2>/dev/null; exit 99; fi
  sleep 10
done
wait $pid; rc=$?; echo "BUILD EXIT $rc" >> oc-build.log; exit $rc
