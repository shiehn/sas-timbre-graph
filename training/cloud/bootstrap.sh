#!/usr/bin/env bash
# Timbre Graph — RunPod pod bootstrap (Ubuntu 22.04/24.04, CPU or GPU pod).
#
#   curl -fsSL https://raw.githubusercontent.com/shiehn/sas-timbre-graph/main/training/cloud/bootstrap.sh | bash
#
# Installs Surge XT (same version as the dev Mac), the lab, and its venv.
# Idempotent: re-running skips work that is already done.
set -euo pipefail

SURGE_VERSION="${SURGE_VERSION:-1.3.4}"
REPO_URL="${REPO_URL:-https://github.com/shiehn/sas-timbre-graph.git}"
WORKDIR="${WORKDIR:-/workspace}"
DEB="surge-xt-linux-x64-${SURGE_VERSION}.deb"
DEB_URL="https://github.com/surge-synthesizer/releases-xt/releases/download/${SURGE_VERSION}/${DEB}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# Surge XT's VST3 links X11/GL/audio libs even when hosted headless; xvfb is
# the fallback if the plugin ever insists on a display.
apt-get install -y -qq --no-install-recommends \
  curl ca-certificates git xz-utils \
  libgl1 libglu1-mesa libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 \
  libxinerama1 libxcomposite1 libxcb1 libxkbcommon-x11-0 libfreetype6 \
  libfontconfig1 libasound2t64 libjack-jackd2-0 xvfb \
  >/dev/null 2>&1 || apt-get install -y -qq --no-install-recommends \
  curl ca-certificates git xz-utils \
  libgl1 libglu1-mesa libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 \
  libxinerama1 libxcomposite1 libxcb1 libxkbcommon-x11-0 libfreetype6 \
  libfontconfig1 libasound2 libjack-jackd2-0 xvfb >/dev/null

log "Surge XT ${SURGE_VERSION}"
if [ ! -d "/usr/lib/vst3/Surge XT.vst3" ]; then
  cd /tmp
  curl -fL --retry 3 -o "$DEB" "$DEB_URL"
  # `apt install ./file.deb` resolves the deb's own dependency closure
  apt-get install -y -qq "./${DEB}" >/dev/null
  rm -f "$DEB"
fi
ls -d "/usr/lib/vst3/Surge XT.vst3" >/dev/null && log "VST3 present"
FACTORY=$(ls -d /usr/share/surge-xt/patches_factory 2>/dev/null || true)
if [ -z "$FACTORY" ]; then
  log "Factory content missing from deb — fetching portable content"
  curl -fL --retry 3 -o /tmp/content.tar.gz \
    "https://github.com/surge-synthesizer/releases-xt/releases/download/${SURGE_VERSION}/surge-xt-portable-content-${SURGE_VERSION}.tar.gz"
  mkdir -p /usr/share/surge-xt
  tar xzf /tmp/content.tar.gz -C /usr/share/surge-xt --strip-components=1
  rm -f /tmp/content.tar.gz
fi
echo "factory patches: $(find /usr/share/surge-xt/patches_factory -name '*.fxp' 2>/dev/null | wc -l)"
echo "3rdparty patches: $(find /usr/share/surge-xt/patches_3rdparty -name '*.fxp' 2>/dev/null | wc -l)"

log "uv + repo"
if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$WORKDIR"
cd "$WORKDIR"
[ -d sas-timbre-graph ] || git clone --depth 1 "$REPO_URL" sas-timbre-graph
cd sas-timbre-graph/training

log "Python env"
uv venv --python 3.11 .venv >/dev/null
# Force the CPU torch build FIRST. On Linux, PyPI's default torch drags in
# the whole CUDA stack (nvidia-* wheels, several GB downloaded and unpacked)
# which is pure waste here: this job renders on CPU and trains in ~11 s.
# Installing it up front satisfies torch>=2.1 so the next resolve leaves it.
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch >/dev/null
uv pip install -e ".[dev]" --python .venv/bin/python >/dev/null
.venv/bin/python -c "import torch; print('torch', torch.__version__, torch.__file__.split('/site-packages/')[0])"
du -sh .venv | sed 's/^/venv size: /'

cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export TGLAB_SURGE_VST3="/usr/lib/vst3/Surge XT.vst3"
export TGLAB_SURGE_CONTENT="/usr/share/surge-xt"
EOF
export TGLAB_SURGE_VST3="/usr/lib/vst3/Surge XT.vst3"
export TGLAB_SURGE_CONTENT="/usr/share/surge-xt"

log "Smoke test: load Surge and render one probe"
.venv/bin/python - <<'PY'
from timbre_graph_lab.config import LabConfig
from timbre_graph_lab.worker import RenderWorker, qc_audio
from timbre_graph_lab.probes import get_probe
import numpy as np, time
cfg = LabConfig()
print("vst3   :", cfg.surge_vst3)
print("factory:", cfg.factory_patches_dir)
t = time.time(); w = RenderWorker(cfg); print(f"host loaded in {time.time()-t:.1f}s")
p = sorted(cfg.factory_patches_dir.rglob("*.fxp"))[0]
t = time.time(); ok = w.load_preset(p); print(f"loaded {p.name} ok={ok} in {time.time()-t:.1f}s")
probe = get_probe("lead", "short")
w.render(probe)
t = time.time(); a = w.render(probe); dt = time.time()-t
print(f"render {dt*1000:.0f} ms  rms={float(np.sqrt(np.mean(a**2))):.4f}  qc={qc_audio(a).ok}")
PY

log "READY"
cat <<EOF

  cd $WORKDIR/sas-timbre-graph/training
  source .venv/bin/activate

  nproc = $(nproc)

  Next:
    tglab inventory
    tglab policy
    tglab parity --write /tmp/parity-linux.json --compare /workspace/parity-macos.json
    tglab gen --per-role 40 --workers \$(( \$(nproc) - 2 ))
    tglab train --epochs 80

EOF
