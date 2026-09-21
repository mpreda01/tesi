#!/bin/bash
# One-time environment setup for the DISI GPU cluster. Run it ON giano.cs.unibo.it (the login machine):
#
#     bash setup_cluster.sh
#
# Everything heavy goes to /scratch.hpc/<user> because the home quota is only 400 MB. Nothing here needs
# root. It is safe to re-run: finished steps are skipped.
#
# Why on giano and not inside the job: the compute nodes may have no internet access, so every download
# (packages, LIBERO-plus assets, model weights) happens here, once.
set -eo pipefail

export SCRATCH_ROOT="/scratch.hpc/matteo.preda/tesi"   # check that this is really your scratch folder
VENV="$SCRATCH_ROOT/venv"
LP="$SCRATCH_ROOT/LIBERO-plus"
export LIBERO_CONFIG_PATH="$SCRATCH_ROOT/libero_config"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export TMPDIR="$SCRATCH_ROOT/tmp"
export MPLCONFIGDIR="$SCRATCH_ROOT/.mplconfig"
export PIP_NO_CACHE_DIR=1                                    # pip's default cache would fill the 400 MB quota
export UV_CACHE_DIR="$SCRATCH_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_ROOT/python"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

mkdir -p "$SCRATCH_ROOT"/{logs,results,tmp,smolvla_eval} "$HF_HOME" "$LIBERO_CONFIG_PATH"

echo "== 1/7  Python >= 3.12 (lerobot 0.6.1 requires it) =="
PY=""
for c in python3.13 python3.12 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  echo "   no system Python >= 3.12: fetching one with uv into $UV_PYTHON_INSTALL_DIR"
  [ -x "$SCRATCH_ROOT/uvtool/bin/uv" ] || { python3 -m venv "$SCRATCH_ROOT/uvtool" && "$SCRATCH_ROOT/uvtool/bin/pip" install uv; }
  "$SCRATCH_ROOT/uvtool/bin/uv" python install 3.12
  PY="$("$SCRATCH_ROOT/uvtool/bin/uv" python find 3.12)"
fi
echo "   using $PY ($("$PY" --version))"

echo "== 2/7  virtualenv + PyTorch (cu118, as the cluster instructions require) =="
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip
# lerobot needs torch>=2.7 and 2.7.1 is the newest release that still ships cu118 wheels.
"$PIP" install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
# torchcodec must match torch (0.5 <-> 2.7). lerobot only imports it lazily for video datasets, so this is
# just to avoid pip picking a build that is ABI-incompatible with torch 2.7.
"$PIP" install "torchcodec==0.5.*"

echo "== 3/7  lerobot + LIBERO-plus runtime dependencies =="
# NOT lerobot[all] / lerobot[libero]: they pull in hf-libero, which shadows LIBERO-plus.
"$PIP" install "lerobot[smolvla,dataset]==0.6.1"
"$PIP" install robosuite==1.4.1 bddl==1.0.1 easydict==1.13 mujoco==3.7.0 \
               Wand==0.6.13 scikit-image==0.25.2 gym==0.26.2 cloudpickle \
               h5py imageio matplotlib scipy pyyaml
# robosuite drags in the GUI build of OpenCV, which needs libGL.so.1 and fails to import on headless nodes.
"$PIP" uninstall -y opencv-python opencv-python-headless
"$PIP" install "opencv-python-headless>=4.9,<4.14"

echo "== 4/7  LIBERO-plus code =="
[ -d "$LP/.git" ] || git clone https://github.com/sylvestf/LIBERO-plus.git "$LP"
echo "   commit: $(git -C "$LP" rev-parse --short HEAD)"

echo "== 5/7  LIBERO-plus assets (6.4 GB zip, ~13 GB extracted; needs ~20 GB free in scratch) =="
"$VENV/bin/python" - "$LP/libero/libero/assets" "$SCRATCH_ROOT/tmp/libero-plus-dl" <<'PY'
import os, shutil, sys, zipfile
from huggingface_hub import hf_hub_download

dst, dl = sys.argv[1], sys.argv[2]
if os.path.isdir(os.path.join(dst, "scenes")):
    print("   assets already in place")
    raise SystemExit
zip_path = hf_hub_download(repo_id="Sylvest/LIBERO-plus", repo_type="dataset", filename="assets.zip", local_dir=dl)
extract = os.path.join(dl, "extract")
print("   extracting...")
zipfile.ZipFile(zip_path).extractall(extract)          # python zipfile: no dependency on the `unzip` binary
# The real assets/ folder sits under a deep author-internal prefix, so find it by content.
found = next((r for r, d, _ in os.walk(extract)
              if os.path.basename(r) == "assets" and ("scenes" in d or "textures" in d)), None)
assert found, "could not locate assets/ inside assets.zip"
shutil.move(found, dst)
shutil.rmtree(dl, ignore_errors=True)
print("   done:", sorted(os.listdir(dst)))
PY

echo "== 6/7  model weights, cached for offline use on the compute nodes =="
"$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
# The SmolVLA checkpoint config points at this VLM backbone and loads its weights/tokenizer at startup.
for repo in ("HuggingFaceVLA/smolvla_libero", "HuggingFaceTB/SmolVLM2-500M-Instruct"):
    print("   ", repo, "->", snapshot_download(repo))
PY

echo "== 7/7  MagickWand check + import check =="
# LIBERO-plus imports `wand` at module level (only its image-blur perturbations use it). Wand needs the
# ImageMagick shared library, which needs root to install. If it is missing we create a stub package that
# is enough for the language perturbation (no image blur is applied there); the notebook refuses to run
# a task that would actually need the real library.
rm -rf "$SCRATCH_ROOT/stubs"
if ! "$VENV/bin/python" -c "import wand.api" 2>/dev/null; then
  echo "   WARNING: MagickWand not found -> creating a stub 'wand' (fine for the text perturbation only)"
  echo "            (for the image perturbations ask DISI to install libmagickwand, or use conda-forge imagemagick)"
  mkdir -p "$SCRATCH_ROOT/stubs/wand"
  : > "$SCRATCH_ROOT/stubs/wand/__init__.py"
  cat > "$SCRATCH_ROOT/stubs/wand/api.py" <<'PY'
class _Fn:
    argtypes = None
    restype = None
    def __call__(self, *a, **k):
        raise RuntimeError("MagickWand is not installed (stub package)")

class _Library:
    def __getattr__(self, name):
        return _Fn()

library = _Library()
PY
  printf 'class Image:\n    pass\n' > "$SCRATCH_ROOT/stubs/wand/image.py"
fi

# Same config the notebook writes; needed here so that `import libero` does not prompt for input().
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LP/libero/libero
bddl_files: $LP/libero/libero/bddl_files
init_states: $LP/libero/libero/init_files
datasets: $LP/libero/datasets
assets: $LP/libero/libero/assets
EOF

EXTRA=""; [ -d "$SCRATCH_ROOT/stubs/wand" ] && EXTRA="$SCRATCH_ROOT/stubs:"
PYTHONPATH="$EXTRA$LP:${PYTHONPATH:-}" "$VENV/bin/python" - <<'PY'
import contextlib, importlib, io
bad = []
for m in ["PIL", "bddl", "cloudpickle", "cv2", "easydict", "gym", "h5py", "huggingface_hub", "imageio", "matplotlib",
          "mujoco", "numpy", "robosuite", "scipy", "skimage", "termcolor", "torch", "tqdm", "wand.api", "yaml", "lerobot"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    raise SystemExit("MISSING IMPORTS:\n  " + "\n  ".join(bad))
import libero.libero.benchmark as benchmark
with contextlib.redirect_stdout(io.StringIO()):          # the constructor prints a 2402-int list
    n = benchmark.get_benchmark_dict()["libero_spatial"]().get_num_tasks()
assert n == 2402, f"libero_spatial has {n} tasks, expected 2402 (is stock LIBERO shadowing LIBERO-plus?)"
print(f"   all imports OK, LIBERO-plus libero_spatial = {n} tasks")
PY

echo
echo "Setup finished. Next, from the folder that contains run_eval.sbatch:"
echo "   sbatch --export=ALL,MAX_TASKS=2 run_eval.sbatch      # 2-task test first"
