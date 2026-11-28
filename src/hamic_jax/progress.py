"""Single-line notebook and terminal progress bars without widgets."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

__all__ = ["ProgressBar"]


def _in_notebook_kernel() -> bool:
    """Return True when running inside an IPython/Jupyter notebook kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


class ProgressBar:
    """Plain-text progress bar that updates one ``display_id`` slot in notebooks.

    Colab and Kaggle headless runners split ``stdout``/``stderr`` or break ANSI
    cursor-up escapes when nested ``tqdm`` bars are active. Inside an IPython
    kernel this renders ``tqdm.format_meter`` into a single ``DisplayHandle`` via
    ``IPython.display.Pretty`` and updates it in-place without widgets.
    Outside notebooks it delegates to standard ``tqdm`` on ``stdout``.
    """

    def __init__(
        self,
        *,
        total: int | float,
        initial: int | float = 0,
        desc: str = "",
        unit: str = "it",
        unit_scale: bool = False,
        leave: bool = True,
        mininterval: float = 0.15,
        postfix: str = "",
    ) -> None:
        self._total = total
        self._initial = initial
        self._n: int | float = initial
        self._desc = desc
        self._unit = unit
        self._unit_scale = unit_scale
        self._leave = leave
        self._mininterval = max(0.0, float(mininterval))
        self._postfix = postfix
        self._start = time.monotonic()
        self._last_render = 0.0
        self._closed = False
        self._lock = threading.Lock()
        self._handle: Any = None
        self._pretty_cls: Any = None
        self._format_meter: Any = None
        self._tqdm: Any = None

        from tqdm import tqdm

        if _in_notebook_kernel():
            from IPython.display import Pretty, display

            self._pretty_cls = Pretty
            self._format_meter = tqdm.format_meter
            self._handle = display(Pretty(self._format()), display_id=True)
            self._last_render = time.monotonic()
        else:
            self._tqdm = tqdm(
                total=total,
                initial=initial,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                leave=leave,
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=mininterval,
            )
            if postfix:
                self._tqdm.set_postfix_str(postfix, refresh=False)

    def _format(self) -> str:
        elapsed = max(time.monotonic() - self._start, 1e-9)
        return str(
            self._format_meter(
                self._n,
                self._total,
                elapsed,
                prefix=self._desc,
                unit=self._unit,
                unit_scale=self._unit_scale,
                bar_format="{l_bar}{bar:20}{r_bar}",
                postfix=self._postfix or None,
                initial=self._initial,
            )
        )

    def _render(self, *, force: bool = False) -> None:
        if self._handle is None or self._closed:
            return
        now = time.monotonic()
        completed = self._total > 0 and self._n >= self._total
        if not force and not completed and (now - self._last_render) < self._mininterval:
            return
        self._handle.update(self._pretty_cls(self._format()))
        self._last_render = now

    def update(self, n: int | float = 1) -> None:
        """Advance the progress counter by ``n``."""
        with self._lock:
            self._n += n
            if self._tqdm is not None:
                self._tqdm.update(n)
            else:
                self._render()

    def set_postfix(
        self,
        ordered_dict: Mapping[str, Any] | None = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        """Set structured key/value postfix fields."""
        items: dict[str, Any] = {}
        if ordered_dict:
            items.update(ordered_dict)
        items.update(kwargs)
        text = ", ".join(f"{key}={value}" for key, value in items.items())
        with self._lock:
            self._postfix = text
            if self._tqdm is not None:
                self._tqdm.set_postfix(items, refresh=refresh)
            elif refresh:
                self._render(force=True)

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """Set a raw postfix string."""
        with self._lock:
            self._postfix = s
            if self._tqdm is not None:
                self._tqdm.set_postfix_str(s, refresh=refresh)
            elif refresh:
                self._render(force=True)

    def write(self, message: str) -> None:
        """Emit a log line cleanly alongside the active progress bar."""
        if self._tqdm is not None:
            self._tqdm.write(message, file=sys.stdout)
        else:
            print(message, flush=True)

    def close(self) -> None:
        """Finalize or clear the progress bar display."""
        with self._lock:
            if self._closed:
                return
            if self._tqdm is not None:
                self._closed = True
                self._tqdm.close()
                return
            if self._handle is not None:
                if self._leave:
                    self._render(force=True)
                else:
                    self._handle.update(self._pretty_cls(""))
            self._closed = True

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
