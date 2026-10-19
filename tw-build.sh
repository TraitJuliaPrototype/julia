#!/bin/bash
# Cold build of the patched 1.13.2, niced, -j4, with a memory watchdog (kill below 4 GB available).
cd "$(dirname "$0")"
setsid nice -n 10 make -j4 ${TW_TARGET:-} > tw-build.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt 4096 ]; then echo "WATCHDOG: MemAvailable ${avail} MB, killing build" >> tw-build.log; kill -TERM -$pid; sleep 5; kill -KILL -$pid 2>/dev/null; exit 99; fi
  sleep 10
done
wait $pid; rc=$?; echo "BUILD EXIT $rc" >> tw-build.log; exit $rc
