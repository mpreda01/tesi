"""SmolVLA on LIBERO-plus: language (text) perturbation.

Evaluates `HuggingFaceVLA/smolvla_libero` on the "Language Instructions" tasks of one LIBERO-plus suite and writes
a single `results.json`:

  * `episodes`: one entry per evaluated episode with `task_id`, the `instruction` the policy received, and `success`;
  * `summary`:  overall accuracy (with a 95% Wilson interval) plus a breakdown by difficulty level and by base task.

No videos are recorded. It runs headless on the DISI GPU cluster through `run_eval.sbatch`, after `setup_cluster.sh` has
prepared the environment once:

    python -u smolvla_text_perturbation.py

Every setting can be overridden with an environment variable (see the configuration block below and `run_eval.sbatch`).
The evaluation resumes from the last finished chunk if the job is killed and restarted with the same settings.
The `# %%` markers split the file into cells, so it can also be run interactively (VS Code, Spyder, Jupytext).
"""

# %%
import contextlib, importlib.metadata, io, json, math, os, platform, subprocess, sys, textwrap, time
from datetime import datetime
from pathlib import Path

# ---- Configuration: every value can be overridden with an environment variable (see cluster/run_eval.sbatch) ----
SCRATCH = Path(os.environ.get("SCRATCH_ROOT", "/scratch.hpc/matteo.preda/tesi"))   # keep in sync with setup_cluster.sh
LP      = Path(os.environ.get("LIBERO_PLUS_DIR", SCRATCH / "LIBERO-plus"))   # clone of github.com/sylvestf/LIBERO-plus
LP_ROOT = LP / "libero" / "libero"                                           # the inner libero/libero folder
HERE    = Path(os.environ.get("PROJECT_DIR")                                 # folder with run_lerobot_eval_novideo.py
               or (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))

SUITE      = os.environ.get("SUITE", "libero_spatial")
POLICY     = os.environ.get("POLICY_PATH", "HuggingFaceVLA/smolvla_libero")
MAX_TASKS  = int(os.environ.get("MAX_TASKS", "0")) or None          # 0 / unset = every language task of the suite
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "25"))                # tasks per lerobot-eval call (= granularity of resume)
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "1"))   # LIBERO-plus protocol: 1 episode per task
SEED       = int(os.environ.get("SEED", "1000"))
N_ACTION_STEPS = os.environ.get("N_ACTION_STEPS", "")               # "" = keep the checkpoint's own value

RUN_NAME = os.environ.get("RUN_NAME", f"smolvla_lang_{SUITE}")
OUT_DIR  = Path(os.environ.get("OUTPUT_DIR", SCRATCH / "results")) / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"suite={SUITE} policy={POLICY}\nout={OUT_DIR}")

# %%
# ---- environment for this process AND for the lerobot-eval child processes ----
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HOME", str(SCRATCH / "hf_cache"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(SCRATCH / "libero_config"))

# `import libero` must resolve to the LIBERO-plus clone (its libero/ folder has no __init__.py, so `pip install -e`
# alone does not make it importable). A stub `wand` exists only if setup_cluster.sh found no MagickWand library.
paths = ([str(SCRATCH / "stubs")] if (SCRATCH / "stubs" / "wand").is_dir() else []) + [str(LP)]
sys.path[:0] = [p for p in paths if p not in sys.path]
existing = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = ":".join(paths + ([existing] if existing else []))

# LIBERO reads its paths from <LIBERO_CONFIG_PATH>/config.yaml and calls input() if it is missing.
cfg_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
cfg_dir.mkdir(parents=True, exist_ok=True)
(cfg_dir / "config.yaml").write_text(textwrap.dedent(f"""\
    benchmark_root: {LP_ROOT}
    bddl_files: {LP_ROOT}/bddl_files
    init_states: {LP_ROOT}/init_files
    datasets: {LP_ROOT}/../datasets
    assets: {LP_ROOT}/assets
"""))

# %%
# ---- fail fast, with a clear message, before spending time on the model ----
import torch

assert torch.cuda.is_available(), (
    "no CUDA GPU visible: was the job submitted with --gres=gpu:1, and is torch the cu118 build?")
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

import libero.libero.benchmark as benchmark

assert str(LP) in benchmark.__file__, f"`libero` resolves to {benchmark.__file__}, not to the LIBERO-plus clone"
assert (LP_ROOT / "assets" / "scenes").is_dir(), f"LIBERO-plus assets missing in {LP_ROOT / 'assets'} (run setup_cluster.sh)"

SUITE_SIZES = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}
assert SUITE in SUITE_SIZES, f"SUITE must be one of {list(SUITE_SIZES)}"
with contextlib.redirect_stdout(io.StringIO()):      # the constructor prints a list with thousands of task indices
    suite = benchmark.get_benchmark_dict()[SUITE]()
assert suite.get_num_tasks() == SUITE_SIZES[SUITE], (
    f"{SUITE} has {suite.get_num_tasks()} tasks, expected {SUITE_SIZES[SUITE]}: stock LIBERO is shadowing LIBERO-plus")
print(f"{SUITE}: {suite.get_num_tasks()} tasks")

# %%
# ---- pick the language-perturbation tasks ----
cls = json.load(open(LP_ROOT / "benchmark" / "task_classification.json"))[SUITE]
lang = [e for e in cls if e["category"] == "Language Instructions"]
if MAX_TASKS:
    lang = lang[:MAX_TASKS]

task_info = {}      # task_id -> the fields that end up in results.json
for e in lang:
    tid = e["id"] - 1                       # task_classification.json ids are 1-based, LeRobot's task_id is 0-based
    t = suite.get_task(tid)
    assert t.name == e["name"], f"id mapping broken for json id {e['id']}: {t.name} != {e['name']}"
    # The `wand` stub (if any) cannot do image blur, and only *_noise_* tasks use it. Language tasks never do.
    assert "_noise_" not in t.name, f"{t.name} needs the real MagickWand library"
    task_info[tid] = {
        "task_id": tid,
        "task_name": t.name,
        "instruction": t.language,          # exactly the string LeRobot hands to the policy
        "base_task": t.name.split("_language_")[0],
        "difficulty_level": e["difficulty_level"],
    }
task_ids = sorted(task_info)
print(f"{len(task_ids)} language tasks in {SUITE} (of {len(cls)} total); first ids: {task_ids[:5]}")
for tid in task_ids[:3]:
    print(f"  {tid}: {task_info[tid]['instruction']}")

# %%
# ---- can this node render? (EGL problems otherwise surface deep inside lerobot-eval) ----
from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_ids[0]), camera_heights=256, camera_widths=256)
try:
    img = env.reset()["agentview_image"]
    assert img.shape == (256, 256, 3) and img.std() > 1, "the renderer returned an empty image (EGL problem?)"
finally:
    env.close()
print("preflight OK: EGL rendering works")

# %%
CHUNK_ROOT = OUT_DIR / "chunks"


def wilson_ci_pct(k, n, z=1.96):
    """95% Wilson score interval for a success rate, in percent."""
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


def accuracy_block(eps):
    n, k = len(eps), sum(e["success"] for e in eps)
    return {"n_episodes": n, "n_success": k, "accuracy_pct": round(100 * k / n, 2) if n else None}


def group_by(eps, key):
    groups = {}
    for e in eps:
        groups.setdefault(str(e[key]), []).append(e)
    return {name: accuracy_block(g) for name, g in sorted(groups.items())}


def _try(fn):
    try:
        return fn()
    except Exception:
        return None


def write_results():
    """(Re)build results.json from every finished chunk. Called after each chunk, so a killed job keeps its progress."""
    episodes, eval_seconds = [], 0.0
    for f in sorted(CHUNK_ROOT.glob("chunk_*/eval_info.json")):
        info = json.load(open(f))
        eval_seconds += info.get("overall", {}).get("eval_s", 0.0)
        for t in info["per_task"]:
            ti = task_info[t["task_id"]]
            for k, ok in enumerate(t["metrics"]["successes"]):
                episodes.append({
                    "task_id": ti["task_id"],
                    "episode_ix": k,
                    "instruction": ti["instruction"],
                    "task_name": ti["task_name"],
                    "base_task": ti["base_task"],
                    "difficulty_level": ti["difficulty_level"],
                    "success": bool(ok),
                })
    episodes.sort(key=lambda e: (e["task_id"], e["episode_ix"]))
    n, k = len(episodes), sum(e["success"] for e in episodes)
    tasks_done = len({e["task_id"] for e in episodes})

    results = {
        "meta": {
            "policy": POLICY,
            "suite": SUITE,
            "perturbation": "Language Instructions",
            "seed": SEED,
            "episodes_per_task": EPISODES_PER_TASK,
            "n_action_steps": N_ACTION_STEPS or "checkpoint default",
            "videos_saved": False,
            "tasks_requested": len(task_ids),
            "tasks_evaluated": tasks_done,
            "complete": tasks_done == len(task_ids),
            "eval_seconds_total": round(eval_seconds, 1),
            "lerobot_version": _try(lambda: importlib.metadata.version("lerobot")),
            "libero_plus_commit": _try(lambda: subprocess.check_output(
                ["git", "-C", str(LP), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()),
            "gpu": _try(lambda: torch.cuda.get_device_name(0)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": platform.node(),
            "written_at": datetime.now().isoformat(timespec="seconds"),
        },
        "episodes": episodes,
        "summary": {
            "n_episodes": n,
            "n_success": k,
            "accuracy": round(k / n, 4) if n else None,
            "accuracy_pct": round(100 * k / n, 2) if n else None,
            "ci95_wilson_pct": wilson_ci_pct(k, n),
            "by_difficulty_level": group_by(episodes, "difficulty_level"),
            "by_base_task": group_by(episodes, "base_task"),
        },
    }
    tmp = OUT_DIR / "results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, OUT_DIR / "results.json")       # atomic: a kill mid-write cannot leave a truncated file
    return results

# %%
# ---- run lerobot-eval chunk by chunk (one fresh process per chunk, no videos) ----
launcher = HERE / "run_lerobot_eval_novideo.py"
assert launcher.is_file(), f"{launcher} not found (set PROJECT_DIR to the folder that contains it)"

# Refuse to mix chunks produced with different settings under one RUN_NAME.
run_cfg = {"policy": POLICY, "suite": SUITE, "seed": SEED, "episodes_per_task": EPISODES_PER_TASK,
           "n_action_steps": N_ACTION_STEPS, "chunk_size": CHUNK_SIZE}
cfg_file = OUT_DIR / "run_config.json"
if cfg_file.is_file():
    prev = json.load(open(cfg_file))
    assert prev == run_cfg, f"{cfg_file} holds a different configuration:\n  {prev}\nvs\n  {run_cfg}\nSet a new RUN_NAME."
else:
    cfg_file.write_text(json.dumps(run_cfg, indent=2))

chunks = [task_ids[i:i + CHUNK_SIZE] for i in range(0, len(task_ids), CHUNK_SIZE)]
t0, finished_now = time.time(), 0
for ci, ids in enumerate(chunks):
    cdir = CHUNK_ROOT / f"chunk_{ci:04d}"
    ids_file = cdir / "task_ids.json"
    if (cdir / "eval_info.json").is_file():
        assert json.load(open(ids_file)) == ids, f"{cdir} was produced for other tasks; set a new RUN_NAME"
        print(f"[chunk {ci + 1}/{len(chunks)}] already done, skipping", flush=True)
        continue
    # With videos off nothing else creates the output dir, and lerobot-eval writes eval_info.json into it at the end.
    cdir.mkdir(parents=True, exist_ok=True)
    ids_file.write_text(json.dumps(ids))

    cmd = [sys.executable, str(launcher),
           f"--policy.path={POLICY}",
           "--env.type=libero_plus", f"--env.task={SUITE}",
           f"--env.task_ids={json.dumps(ids, separators=(',', ':'))}",
           f"--eval.n_episodes={EPISODES_PER_TASK}", "--eval.batch_size=1",
           "--env.max_parallel_tasks=1",     # tasks share one policy object (stateful action queue): never run them in threads
           f"--seed={SEED}", f"--output_dir={cdir}"]
    if N_ACTION_STEPS:
        cmd.append(f"--policy.n_action_steps={N_ACTION_STEPS}")

    print(f"[chunk {ci + 1}/{len(chunks)}] {len(ids)} tasks (ids {ids[0]}..{ids[-1]})", flush=True)
    t_chunk = time.time()
    with open(cdir / "lerobot_eval.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "".join(open(cdir / "lerobot_eval.log", errors="replace").readlines()[-40:])
        raise RuntimeError(f"lerobot-eval failed on chunk {ci} (exit {rc}). Last log lines:\n{tail}")

    finished_now += 1
    write_results()
    eta_min = (time.time() - t0) / finished_now * (len(chunks) - ci - 1) / 60
    print(f"    {time.time() - t_chunk:.0f}s | ETA ~{eta_min:.0f} min", flush=True)

results = write_results()
s, m = results["summary"], results["meta"]
print(f"\n=== {POLICY} | {SUITE} | language perturbation ===")
print(f"tasks {m['tasks_evaluated']}/{m['tasks_requested']} | episodes {s['n_episodes']} | "
      f"accuracy {s['accuracy_pct']}%  (95% CI {s['ci95_wilson_pct']})")
for lvl, b in s["by_difficulty_level"].items():
    print(f"  difficulty {lvl}: {b['accuracy_pct']}%  ({b['n_success']}/{b['n_episodes']})")
print("results:", OUT_DIR / "results.json")
