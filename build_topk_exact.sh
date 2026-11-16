#!/bin/bash
# Incremental rebuild of the vLLM CUDA extension for the persistent_topk
# exactness backport (upstream 33a73daf177) on wt-rebase4.
#
# - Uses the tree's existing cmake-build-release/ dir (CMake cache points at
#   /home/jslave2/wt-rebase4, Release, CUDA 13.2) so only the TUs that
#   include the changed top-k headers are recompiled.
# - MAX_JOBS=2: each cicc needs ~4.5 GB RAM; this node has ~14 GB free and
#   a live vLLM server holding ~8 GB. 4 parallel nvcc jobs have OOM-killed
#   this node before.
# - No rsync to the other nodes: the user handles serving deployment.

set -o pipefail

WT=/home/jslave2/wt-rebase4
BUILD=$WT/cmake-build-release
LOG=/tmp/vllm-build-topk-exact.log

export PATH="$HOME/.local/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION="0.31.0rc0.dev0+g7904c4f626"
export VLLM_VERSION_OVERRIDE="$SETUPTOOLS_SCM_PRETEND_VERSION"

# Cache-clear watchdog, same as cluster-ops/build-vllm-wt.sh: the build dir
# and the RAM-heavy compiler both benefit from a page cache flush.
(
  while true; do
    sleep 120
    sync
    sudo sysctl vm.drop_caches=1 >/dev/null 2>&1
  done
) >/dev/null 2>&1 &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

cd "$BUILD" || exit 1

echo "[$(date)] ninja build start (jobs=2) in $BUILD"
START=$(date +%s)

# Build only the target that owns the changed top-k sources.
ninja -j 2 _C_stable_libtorch 2>&1
RC=$?

echo "[$(date)] ninja exit code: $RC"
if [ $RC -ne 0 ]; then
  echo "[$(date)] BUILD FAILED - not installing"
  exit $RC
fi

# Install the built component into the package dir (vllm/).
echo "[$(date)] installing _C_stable_libtorch component"
cmake --install . --prefix "$WT" --component _C_stable_libtorch 2>&1
IRC=$?
echo "[$(date)] install exit code: $IRC"

END=$(date +%s)
echo "[$(date)] total build duration: $((END - START)) s"
ls -la --time-style=full-iso "$WT/vllm/_C_stable_libtorch.abi3.so"
md5sum "$WT/vllm/_C_stable_libtorch.abi3.so"
exit $(( RC != 0 ? RC : IRC ))
