from __future__ import annotations

import contextlib
import contextvars
import logging
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

from tqdm import tqdm

_outer_position: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "trails_outer_position",
    default=None,
)
_inner_position: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "trails_inner_position",
    default=None,
)
_leave_bars: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "trails_leave_bars",
    default=None,
)
_description_prefix: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trails_description_prefix",
    default=None,
)


class CompactTqdmFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()
        return super().format(record)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def configure_tqdm_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    formatter = CompactTqdmFormatter("%(levelname)s:%(name)s:%(message)s")

    for handler in list(root_logger.handlers):
        if isinstance(handler, TqdmLoggingHandler):
            handler.setLevel(level)
            handler.setFormatter(formatter)
            root_logger.setLevel(level)
            _configure_warnings()
            return
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            logging.FileHandler,
        ):
            root_logger.removeHandler(handler)

    handler = TqdmLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    _configure_warnings()


def configure_tqdm_lock(lock: Any) -> None:
    tqdm.set_lock(lock)


def _configure_warnings() -> None:
    logging.captureWarnings(True)
    warnings.filterwarnings(
        "ignore",
        message=".*constrained_layout not applied.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Sparse invariant checks are implicitly disabled.*",
        category=UserWarning,
    )


@contextlib.contextmanager
def progress_context(
    *,
    outer_position: int | None = None,
    inner_position: int | None = None,
    leave: bool | None = None,
    description_prefix: str | None = None,
) -> Iterator[None]:
    outer_token = _outer_position.set(outer_position)
    inner_token = _inner_position.set(inner_position)
    leave_token = _leave_bars.set(leave)
    prefix_token = _description_prefix.set(description_prefix)
    try:
        yield
    finally:
        _description_prefix.reset(prefix_token)
        _leave_bars.reset(leave_token)
        _inner_position.reset(inner_token)
        _outer_position.reset(outer_token)


def progress_bar[T](
    iterable: Iterable[T],
    *,
    desc: str,
    level: str,
    leave: bool | None = None,
    **kwargs: Any,
) -> tqdm[T]:
    if level == "outer":
        position = _outer_position.get()
        default_leave = True
    elif level == "inner":
        position = _inner_position.get()
        default_leave = False
    else:
        raise ValueError(f"Unsupported progress bar level: {level}")

    context_leave = _leave_bars.get()
    resolved_leave = (
        leave
        if leave is not None
        else (context_leave if context_leave is not None else default_leave)
    )
    prefix = _description_prefix.get()
    resolved_desc = desc if prefix is None else f"{prefix} {desc}"
    return tqdm(
        iterable,
        desc=resolved_desc,
        position=position,
        leave=resolved_leave,
        **kwargs,
    )
