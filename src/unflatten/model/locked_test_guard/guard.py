"""RFC005 §23.2-23.3 locked-test one-shot access guard.

State machine (§23.2):

```text
sealed_unopened -> access_reserved -> consumed
```

§23.3: the `sealed_unopened -> access_reserved` transition "must be
atomically written before decrypting/materializing the first test
artefact"; a crash, cancel, timeout, partial result or evaluation-code
failure "must never revert the branch to sealed_unopened; the state
closes as consumed." This guard implements both halves: the reservation
write happens strictly before the caller-supplied `read_fn` runs, and a
`finally` unconditionally closes the state to `consumed` even when
`read_fn` raises. A branch found already `access_reserved` (a prior
process crashed after committing the reservation but before closing it)
is itself closed to `consumed` and refused — it is never left readable
and never reverts to `sealed_unopened`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class LockedTestState(StrEnum):
    SEALED_UNOPENED = "sealed_unopened"
    ACCESS_RESERVED = "access_reserved"
    CONSUMED = "consumed"


class SecondAccessRefusedError(RuntimeError):
    """The branch is not `sealed_unopened`; a second raw-data access is refused."""


class LockedTestGuard:
    """Wraps one locked-test branch's atomic state file at `state_path`."""

    def __init__(self, state_path: Path) -> None:
        self._state_path = state_path

    @classmethod
    def initialize(cls, state_path: Path) -> LockedTestGuard:
        """Create a fresh `sealed_unopened` state file. Refuses to overwrite."""
        if state_path.exists():
            raise FileExistsError(f"{state_path} already exists; refusing to reinitialize")
        guard = cls(state_path)
        guard._write_state(LockedTestState.SEALED_UNOPENED)
        return guard

    def read_state(self) -> LockedTestState:
        return LockedTestState(self._state_path.read_text().strip())

    def _write_state(self, state: LockedTestState) -> None:
        """Tmp + fsync + `os.replace` — never a partial write is observable."""
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with temporary.open("w") as handle:
            handle.write(state.value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._state_path)
        self._fsync_parent_dir()

    def _fsync_parent_dir(self) -> None:
        directory = self._state_path.parent
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def reserve_and_read(self, read_fn: Callable[[], T]) -> T:
        """Reserve the branch, run `read_fn`, close to `consumed`.

        Raises :class:`SecondAccessRefusedError` if the branch is not
        `sealed_unopened` — including a stale `access_reserved` left by a
        crashed prior attempt, which this call closes to `consumed` as
        part of refusing it, per §23.3.
        """
        current = self.read_state()
        if current is LockedTestState.ACCESS_RESERVED:
            self._write_state(LockedTestState.CONSUMED)
            raise SecondAccessRefusedError(
                f"{self._state_path} was left access_reserved by a prior attempt; "
                "closed to consumed and refusing this access"
            )
        if current is LockedTestState.CONSUMED:
            raise SecondAccessRefusedError(f"{self._state_path} is already consumed")

        self._write_state(LockedTestState.ACCESS_RESERVED)
        try:
            return read_fn()
        finally:
            self._write_state(LockedTestState.CONSUMED)
