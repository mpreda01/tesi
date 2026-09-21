"""lerobot-eval, with rollout videos disabled.

lerobot 0.6.1 always renders (and encodes) up to 10 rollout videos per task, and offers no CLI flag to turn
that off: `eval_main` hard-codes `max_episodes_rendered = 0 if cfg.eval.recording else 10`. This wrapper
forces it to 0 and drops the videos dir, then runs the stock `lerobot-eval` entry point unchanged, so every
CLI argument works exactly as documented.

Side effect: with no videos dir nothing creates `--output_dir` any more, so the caller must create it
before launching (the notebook does).

Usage: python run_lerobot_eval_novideo.py --policy.path=... --env.type=libero_plus ...
"""

import lerobot.scripts.lerobot_eval as _lerobot_eval

_orig_eval_policy_all = _lerobot_eval.eval_policy_all


def _eval_policy_all_no_video(*args, **kwargs):
    kwargs["max_episodes_rendered"] = 0
    kwargs["videos_dir"] = None
    return _orig_eval_policy_all(*args, **kwargs)


# `eval_main` looks `eval_policy_all` up in the module globals at call time, so this rebinding takes effect.
_lerobot_eval.eval_policy_all = _eval_policy_all_no_video

if __name__ == "__main__":
    _lerobot_eval.main()
