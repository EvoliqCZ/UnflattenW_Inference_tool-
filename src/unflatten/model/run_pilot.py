"""Promotion pilot harness (DEC-049 point 8): every run longer than a day
gets a short (1-2 h) pilot on the final cache and final code before it may
start -- loading, first steps, worst-page memory, checkpoint and resume.

Generic over the trainer's own CLI `main(argv) -> int`. Built first for
`s4_train.main` (the dense fallback), so the interface (`PilotConfig`, the
`runner`/`memory_probe`/`forward_backward`/`loss_reader` hooks) is designed
to let `s6_train` plug in later WITHOUT editing `s6_train` itself: a caller
only ever supplies new hook implementations, never a code change here.

Repo lessons this encodes directly (see `docs/decisions/DEC-049-...md`
point 8 and the memory index it draws on): the first v15b promotion start
failed on things a dry run did not reach (aligner state, an OOM on the
worst page, a restore race) -- "replay step 6-7 + worst-page memory before
launch"; long runs need progress + resume + unbuffered logs; a warm start
must log which keys were freshly initialised.

Five independent checks, each pass/fail on its own evidence -- a check
that cannot run (no hook configured, no CUDA fact to compare, log
missing) is a FAILURE of that check, never a silent skip that inflates
the verdict:

- `pilot_run`: the trainer accepts a small step budget and exits 0, and
  its own step ledger is non-empty and strictly increasing.
- `resume_continuity`: a second attempt, resumed from the pilot's own
  mid-run checkpoint, extends the SAME ledger file strictly increasing
  with no duplicate step; if a `loss_reader` is configured, the resumed
  run's loss at the final step must equal the uninterrupted run's.
- `worst_page_memory`: one forced forward+backward on the cache's
  largest page (found from row metadata, not by loading every row's
  features into a batch) stays under the configured memory budget.
  `peak_memory_bytes` is `None` off CUDA -- that is accepted as-is, per
  design, never treated as a failure to divide by.
- `unbuffered_log`: the pilot run's own log was written through a
  line-buffered / `-u` channel, not batched.
- `fresh_keys_logged`: the pilot run's log named which state_dict keys a
  warm start freshly initialised (never silently assumed empty).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from unflatten.model.exact_text_cache import PageKey

#: `s2_train._LEDGER_FILENAME` / `s4_train._LEDGER_FILENAME` verbatim --
#: the shared convention every `s{2,3,4}_train` ledger file uses.
DEFAULT_LEDGER_FILENAME = "exposure_ledger.jsonl"

#: The default regex a pilot's log must contain for `fresh_keys_logged`
#: to pass: `InitReport.fresh_keys` is always printed under this name
#: (`s4_train.py`'s own admission-report line), whether it names zero or
#: more keys.
DEFAULT_FRESH_KEYS_PATTERN = r"fresh_keys"


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What one `runner` invocation produced: the trainer's own exit
    code, the log file it was captured into, and whether that capture
    channel was actually unbuffered -- a fact the runner derives from
    what it did, never a hardcoded `True`."""

    returncode: int
    log_path: Path
    launched_with_unbuffered: bool
    command: tuple[str, ...] = ()


RunnerFn = Callable[[list[str], Path], RunResult]


def subprocess_runner(module: str) -> RunnerFn:
    """A `RunnerFn` that launches the trainer as `python -u -m <module>
    <argv>` with `PYTHONUNBUFFERED=1` set, stdout+stderr captured into
    `log_path`. `launched_with_unbuffered` is computed FROM the command
    actually built, so a caller who strips `-u` (or the env var) sees the
    `unbuffered_log` check fail instead of trusting a stale flag."""

    def _run(argv: list[str], log_path: Path) -> RunResult:
        cmd = [sys.executable, "-u", "-m", module, *argv]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(log_path, "w", buffering=1) as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
            )
        return RunResult(
            returncode=proc.returncode,
            log_path=log_path,
            launched_with_unbuffered=("-u" in cmd or env.get("PYTHONUNBUFFERED") == "1"),
            command=tuple(cmd),
        )

    return _run


def in_process_runner(trainer_main: Callable[[list[str]], int]) -> RunnerFn:
    """A `RunnerFn` that calls `trainer_main(argv)` directly in this
    process, for tests over a stub trainer. The log is opened with
    `buffering=1` (line-buffered text mode); `launched_with_unbuffered`
    reads `line_buffering` back off that same file object -- a real fact
    about the handle actually used, not an assumption."""

    def _run(argv: list[str], log_path: Path) -> RunResult:
        with open(log_path, "w", buffering=1) as log:
            with contextlib.redirect_stdout(log):
                returncode = trainer_main(argv)
            unbuffered = log.line_buffering
        return RunResult(
            returncode=returncode, log_path=log_path, launched_with_unbuffered=unbuffered
        )

    return _run


def _with_flag(argv: Sequence[str], flag: str, value: str) -> list[str]:
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv = [*argv, flag, value]
    return argv


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return None


def read_ledger_steps(ledger_path: Path, *, step_field: str = "optimizer_step") -> list[int]:
    """Every `step_field` value in `ledger_path`, in file order. `[]` if
    the ledger does not exist yet (a pilot run that never got that far is
    reported by an empty, not-strictly-increasing sequence, never by an
    exception a caller has to special-case)."""
    if not ledger_path.exists():
        return []
    steps = []
    for line in ledger_path.read_text().splitlines():
        if line.strip():
            steps.append(json.loads(line)[step_field])
    return steps


def _strictly_increasing(steps: Sequence[int]) -> bool:
    return all(b > a for a, b in zip(steps, steps[1:]))


def find_worst_page_index(cache_dir: Path) -> tuple[PageKey, int]:
    """The page with the largest visual token count (`grid_thw[1] *
    grid_thw[2]`) in an exact-text cache, WITHOUT building a training
    batch for every row: each shard is opened once to read its rows'
    `document_bundle_id` / `page_index` / `grid_thw` fields, the same
    metadata `exact_text_cache._row_to_storage` writes, and nothing else
    from the row is used. Refuses on an empty cache -- a pilot has
    nothing to probe memory with."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    best_key: PageKey | None = None
    best_size = -1
    for shard in manifest["shard_index"]:
        rows = torch.load(cache_dir / shard["name"], weights_only=True)
        for row in rows:
            grid = row["grid_thw"]
            size = int(grid[1]) * int(grid[2])
            if size > best_size:
                best_size = size
                best_key = (row["document_bundle_id"], int(row["page_index"]))
    if best_key is None:
        raise ValueError(f"{cache_dir}: no pages found in manifest -- nothing to probe")
    return best_key, best_size


@dataclasses.dataclass(frozen=True)
class MemoryProbeResult:
    page_key: str
    peak_memory_bytes: int | None


def probe_worst_page_memory(
    cache_dir: Path,
    *,
    worst_page_and_batch: Callable[[Path], tuple[Any, Any]],
    forward_backward: Callable[[Any], None],
) -> MemoryProbeResult:
    """`worst_page_and_batch` resolves the cache's worst page (typically
    via `find_worst_page_index`) into whatever trainer-specific batch
    object `forward_backward` needs -- both are supplied by the caller so
    this module never has to know a trainer's model/collate internals.
    CUDA peak stats are reset before the forced step and read back after;
    `peak_memory_bytes` is `None` off CUDA, on purpose (design: "None on
    CPU" is an accepted fact, not a probe failure)."""
    page_key, batch = worst_page_and_batch(cache_dir)
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
    forward_backward(batch)
    peak = int(torch.cuda.max_memory_allocated()) if has_cuda else None
    return MemoryProbeResult(page_key=str(page_key), peak_memory_bytes=peak)


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PilotReport:
    verdict: str
    checks: tuple[CheckResult, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def write_report(report: PilotReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True))


@dataclasses.dataclass(frozen=True)
class PilotConfig:
    """Everything one pilot needs. `base_argv` is the trainer's own CLI
    argv, exactly as a real launch would build it (positional cache dir,
    `--run-dir`, `--out`, etc.) -- this harness only ever overrides the
    step/checkpoint/resume flags it names below, never anything else."""

    runner: RunnerFn
    base_argv: tuple[str, ...]
    run_dir: Path
    cache_dir: Path
    log_dir: Path
    pilot_steps: int
    resume_steps: int
    checkpoint_every: int
    worst_page_and_batch: Callable[[Path], tuple[Any, Any]]
    forward_backward: Callable[[Any], None]
    memory_budget_bytes: int
    steps_flag: str = "--steps"
    checkpoint_every_flag: str = "--checkpoint-every"
    out_flag: str = "--out"
    resume_flag: str = "--resume"
    run_dir_flag: str = "--run-dir"
    ledger_filename: str = DEFAULT_LEDGER_FILENAME
    step_field: str = "optimizer_step"
    fresh_keys_pattern: str = DEFAULT_FRESH_KEYS_PATTERN
    loss_reader: Callable[[Path, int], float] | None = None
    uninterrupted_runner: RunnerFn | None = None
    uninterrupted_run_dir: Path | None = None
    loss_tolerance: float = 1e-6


def _check_pilot_run(config: PilotConfig) -> tuple[CheckResult, list[str], RunResult]:
    pilot_argv = _with_flag(
        config.base_argv, config.checkpoint_every_flag, str(config.checkpoint_every)
    )
    pilot_argv = _with_flag(pilot_argv, config.steps_flag, str(config.pilot_steps))
    pilot_log = config.log_dir / "pilot_run.log"
    result = config.runner(pilot_argv, pilot_log)
    ledger_path = config.run_dir / config.ledger_filename
    steps = read_ledger_steps(ledger_path, step_field=config.step_field)
    passed = bool(steps) and result.returncode == 0 and _strictly_increasing(steps)
    check = CheckResult(
        "pilot_run", passed,
        {"returncode": result.returncode, "ledger_steps": steps, "log_path": str(pilot_log)},
    )
    return check, pilot_argv, result


def _check_resume(
    config: PilotConfig, pilot_argv: list[str], pilot_passed: bool
) -> CheckResult:
    if not pilot_passed:
        return CheckResult("resume_continuity", False, {"reason": "pilot_run did not pass"})
    checkpoint = _flag_value(pilot_argv, config.out_flag)
    if checkpoint is None:
        return CheckResult(
            "resume_continuity", False,
            {"reason": f"{config.out_flag} not set in base_argv -- cannot resume"},
        )
    ledger_path = config.run_dir / config.ledger_filename
    steps_before = read_ledger_steps(ledger_path, step_field=config.step_field)
    total_steps = config.pilot_steps + config.resume_steps
    resume_argv = _with_flag(pilot_argv, config.steps_flag, str(total_steps))
    resume_argv = _with_flag(resume_argv, config.resume_flag, checkpoint)
    resume_log = config.log_dir / "resume_run.log"
    result = config.runner(resume_argv, resume_log)
    steps_after = read_ledger_steps(ledger_path, step_field=config.step_field)
    continuity_ok = (
        result.returncode == 0
        and _strictly_increasing(steps_after)
        and len(steps_after) > len(steps_before)
    )
    detail: dict[str, Any] = {
        "returncode": result.returncode,
        "ledger_steps": steps_after,
        "log_path": str(resume_log),
    }
    if continuity_ok and config.loss_reader is not None:
        continuity_ok = _check_resume_determinism(config, total_steps, detail)
    return CheckResult("resume_continuity", continuity_ok, detail)


def _check_resume_determinism(
    config: PilotConfig, total_steps: int, detail: dict[str, Any]
) -> bool:
    """The determinism half of `resume_continuity`: only reached when a
    `loss_reader` is configured (production usage may skip this half
    entirely for a non-deterministic trainer, but a CONFIGURED
    `loss_reader` must actually be honoured, never quietly ignored)."""
    if config.uninterrupted_runner is None or config.uninterrupted_run_dir is None:
        detail["reason"] = "loss_reader configured without an uninterrupted_runner/run_dir"
        return False
    uninterrupted_argv = _with_flag(
        config.base_argv, config.checkpoint_every_flag, str(config.checkpoint_every)
    )
    uninterrupted_argv = _with_flag(uninterrupted_argv, config.steps_flag, str(total_steps))
    uninterrupted_argv = _with_flag(
        uninterrupted_argv, config.run_dir_flag, str(config.uninterrupted_run_dir)
    )
    uninterrupted_argv = _with_flag(
        uninterrupted_argv, config.out_flag,
        str(Path(config.uninterrupted_run_dir) / "checkpoint.json"),
    )
    uninterrupted_log = config.log_dir / "uninterrupted_run.log"
    uninterrupted_result = config.uninterrupted_runner(uninterrupted_argv, uninterrupted_log)
    resumed_loss = config.loss_reader(config.run_dir, total_steps)
    uninterrupted_loss = config.loss_reader(config.uninterrupted_run_dir, total_steps)
    detail["resumed_loss"] = resumed_loss
    detail["uninterrupted_loss"] = uninterrupted_loss
    detail["uninterrupted_returncode"] = uninterrupted_result.returncode
    loss_match = abs(resumed_loss - uninterrupted_loss) <= config.loss_tolerance
    return uninterrupted_result.returncode == 0 and loss_match


def _check_unbuffered_and_fresh_keys(
    pilot_result: RunResult, pattern: str
) -> tuple[CheckResult, CheckResult]:
    unbuffered_check = CheckResult(
        "unbuffered_log",
        pilot_result.launched_with_unbuffered,
        {"log_path": str(pilot_result.log_path)},
    )
    log_text = pilot_result.log_path.read_text() if pilot_result.log_path.exists() else ""
    fresh_keys_ok = re.search(pattern, log_text) is not None
    fresh_keys_check = CheckResult("fresh_keys_logged", fresh_keys_ok, {"pattern": pattern})
    return unbuffered_check, fresh_keys_check


def _check_worst_page_memory(config: PilotConfig) -> CheckResult:
    result = probe_worst_page_memory(
        config.cache_dir,
        worst_page_and_batch=config.worst_page_and_batch,
        forward_backward=config.forward_backward,
    )
    within_budget = (
        result.peak_memory_bytes is None or result.peak_memory_bytes <= config.memory_budget_bytes
    )
    return CheckResult(
        "worst_page_memory", within_budget,
        {
            "page_key": result.page_key,
            "peak_memory_bytes": result.peak_memory_bytes,
            "budget_bytes": config.memory_budget_bytes,
        },
    )


def run_pilot(config: PilotConfig) -> PilotReport:
    """Runs every check in `PilotReport`'s own docstring order and folds
    them into one verdict: `PASS` only if every check passed, `FAIL`
    otherwise -- a check this function could not even attempt (e.g. no
    checkpoint to resume from) is recorded as a failed check, never
    omitted."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    pilot_check, pilot_argv, pilot_result = _check_pilot_run(config)
    unbuffered_check, fresh_keys_check = _check_unbuffered_and_fresh_keys(
        pilot_result, config.fresh_keys_pattern
    )
    resume_check = _check_resume(config, pilot_argv, pilot_check.passed)
    memory_check = _check_worst_page_memory(config)

    checks = (pilot_check, resume_check, memory_check, unbuffered_check, fresh_keys_check)
    verdict = "PASS" if all(c.passed for c in checks) else "FAIL"
    return PilotReport(verdict=verdict, checks=checks)


def _load_entry_point(spec: str) -> Callable[..., Any]:
    """`"package.module:function"` -> the imported function. Used by the
    CLI to let an operator wire a real `worst_page_and_batch` /
    `forward_backward` / `loss_reader` for whichever trainer is being
    piloted, without this module importing any trainer's heavy model
    code itself."""
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"entry point must be 'module:function', got {spec!r}")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-module", required=True, help="run via `python -u -m <module>`")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--pilot-steps", type=int, required=True)
    parser.add_argument("--resume-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--memory-budget-bytes", type=int, required=True)
    parser.add_argument("--worst-page-and-batch", required=True, help="'module:function'")
    parser.add_argument("--forward-backward", required=True, help="'module:function'")
    parser.add_argument("--ledger-filename", default=DEFAULT_LEDGER_FILENAME)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "trainer_argv", nargs=argparse.REMAINDER,
        help="remaining args passed verbatim to the trainer's own CLI",
    )
    args = parser.parse_args(argv)

    config = PilotConfig(
        runner=subprocess_runner(args.trainer_module),
        base_argv=tuple(args.trainer_argv),
        run_dir=args.run_dir,
        cache_dir=args.cache,
        log_dir=args.log_dir,
        pilot_steps=args.pilot_steps,
        resume_steps=args.resume_steps,
        checkpoint_every=args.checkpoint_every,
        worst_page_and_batch=_load_entry_point(args.worst_page_and_batch),
        forward_backward=_load_entry_point(args.forward_backward),
        memory_budget_bytes=args.memory_budget_bytes,
        ledger_filename=args.ledger_filename,
    )
    report = run_pilot(config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, args.report)
    print(json.dumps(report.to_json()))
    return 0 if report.verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
