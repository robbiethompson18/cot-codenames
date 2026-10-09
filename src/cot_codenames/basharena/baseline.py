"""Honest BashArena baseline: can a candidate policy model do the main tasks at all, before any training?

Each model runs the same `--tasks` main tasks (a seeded shuffle of the 640) with BashArena's own honest policy, tools and
100-action cutoff, through OpenRouter. No side task is asked for and no monitor runs; the side-task checkers still run
after every action, so accidental triggers show up in the log. One Inspect log per model lands in `--log-dir`, with
every message, the model's reasoning, token counts and the fraction of hidden tests passed.

Needs Docker with Compose, OPENROUTER_API_KEY, and about 2.3 GB of RAM per trajectory running at once.

    uv run --group basharena python -m cot_codenames.basharena.baseline --models qwen/qwen3.6-27b z-ai/glm-5.3
"""

import argparse

from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.settings.bash_arena.bash_arena_setting import BashArenaSetting
from control_arena.settings.bash_arena.policies.honest_policy import honest_policy
from control_arena.settings.bash_arena.protocol.auto_submit_protocol_wrapper import auto_submit_protocol_wrapper
from control_arena.settings.bash_arena.task.dataset import DatasetSpec
from inspect_ai import eval as inspect_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="OpenRouter model ids, one Inspect task each")
    ap.add_argument("--tasks", type=int, default=50, help="main tasks per model")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed that picks the tasks; the same for every model")
    ap.add_argument("--max-steps", type=int, default=100, help="actions per trajectory (the paper's cutoff)")
    ap.add_argument("--parallel", type=int, default=4, help="trajectories running at once, per model")
    ap.add_argument("--log-dir", default="runs/basharena/baseline")
    args = ap.parse_args()

    setting = BashArenaSetting(dataset_spec=DatasetSpec(shuffle_seed=args.seed, max_tasks=args.tasks))
    tasks = []
    for model in args.models:
        name = f"openrouter/{model}"
        config = ControlEvalConfig(
            max_steps=args.max_steps,
            # The default 1M-token limit per sample is below what a 100-action trajectory re-reading its own context uses.
            token_limit=8_000_000,
            # Both roles point at the policy model so nothing falls back to ControlArena's default OpenAI/Anthropic models.
            untrusted_model=name,
            trusted_model=name,
            max_tasks=len(args.models),
            max_samples=args.parallel,
            log_dir=args.log_dir,
        )
        micro_protocol = auto_submit_protocol_wrapper(micro_protocol=honest_policy(model_name=name))
        tasks.append(get_control_task(setting, micro_protocol, EvalMode.HONEST, config, task_name_prefix=model.replace("/", "-")))
    inspect_eval(tasks, **config.eval_args())


if __name__ == "__main__":
    main()
