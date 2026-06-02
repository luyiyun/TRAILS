from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.handlers
import multiprocessing as mp
import shutil
import textwrap
import warnings
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm


@dataclass(frozen=True)
class _ProgressScope:
    workers: int | None = None
    worker_slot: int | None = None
    description_prefix: str | None = None
    leave: bool | None = None


@dataclass(frozen=True)
class _ProgressNode:
    parent: _ProgressNode | None
    local_row: int

    @classmethod
    def from_active_parent(cls) -> _ProgressNode:
        parent = _active_progress_node.get()
        return cls(parent=parent, local_row=0 if parent is None else parent.local_row + 1)


_active_progress_node: contextvars.ContextVar[_ProgressNode | None] = contextvars.ContextVar(
    "trails_active_progress_node",
    default=None,
)
_progress_scope: contextvars.ContextVar[_ProgressScope | None] = contextvars.ContextVar(
    "trails_progress_scope",
    default=None,
)


class _ProgressLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        scope = _progress_scope.get()
        if scope is not None and scope.description_prefix is not None:
            record.progress_description_prefix = scope.description_prefix
        return True


class CompactTqdmFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            message = record.getMessage()
        elif record.name == "py.warnings":
            message = " ".join(record.getMessage().split())
            message = f"WARNING:{message}"
        else:
            message = super().format(record)
        prefix = getattr(record, "progress_description_prefix", None)
        if isinstance(prefix, str) and prefix:
            message = f"{prefix} {message}"
        return _shorten_log_line(message)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _shorten_log_line(message: str) -> str:
    compact_message = " ".join(message.split())
    return textwrap.shorten(
        compact_message,
        width=_terminal_log_width(),
        placeholder=" ...",
    )


def _terminal_log_width() -> int:
    return max(40, shutil.get_terminal_size(fallback=(120, 20)).columns - 1)


def _make_tqdm_logging_handler(level: int) -> TqdmLoggingHandler:
    handler = TqdmLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(CompactTqdmFormatter("%(levelname)s:%(name)s:%(message)s"))
    _ensure_progress_log_filter(handler)
    return handler


def _ensure_progress_log_filter(handler: logging.Handler) -> None:
    if not any(isinstance(log_filter, _ProgressLogFilter) for log_filter in handler.filters):
        handler.addFilter(_ProgressLogFilter())


def configure_tqdm_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        if isinstance(handler, TqdmLoggingHandler):
            handler.setLevel(level)
            handler.setFormatter(CompactTqdmFormatter("%(levelname)s:%(name)s:%(message)s"))
            _ensure_progress_log_filter(handler)
            root_logger.setLevel(level)
            _configure_warnings()
            return
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            logging.FileHandler,
        ):
            root_logger.removeHandler(handler)

    root_logger.addHandler(_make_tqdm_logging_handler(level))
    root_logger.setLevel(level)
    _configure_warnings()


def _configure_tqdm_queue_logging(log_queue: Any, *, level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(level)
    _ensure_progress_log_filter(queue_handler)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)
    _configure_warnings()


def _configure_warnings() -> None:
    logging.captureWarnings(True)
    # 并行 tqdm 下超长 warning 会折行并留下残影；这条 PyTorch 原型提示不影响训练结果。
    warnings.filterwarnings(
        "ignore",
        message=".*The PyTorch API of nested tensors is in prototype stage.*",
        category=UserWarning,
    )
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


class ProgressBar[T]:
    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        *,
        desc: str | None = None,
        leave: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self._node = _ProgressNode.from_active_parent()
        self._enter_token: contextvars.Token[_ProgressNode | None] | None = None
        position = kwargs.pop("position", None)
        self._bar = tqdm(
            iterable,
            desc=self._description(desc),
            position=self._position() if position is None else position,
            leave=self._leave(leave),
            **kwargs,
        )

    def __iter__(self) -> Iterator[T]:
        token = _active_progress_node.set(self._node)
        try:
            yield from self._bar
        finally:
            _active_progress_node.reset(token)
            self.close()

    def __enter__(self) -> ProgressBar[T]:
        self._enter_token = _active_progress_node.set(self._node)
        enter = getattr(self._bar, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            exit_ = getattr(self._bar, "__exit__", None)
            if exit_ is not None:
                exit_(exc_type, exc, traceback)
            else:
                self.close()
        finally:
            if self._enter_token is not None:
                _active_progress_node.reset(self._enter_token)
                self._enter_token = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bar, name)

    def update(self, n: int | float = 1) -> bool | None:
        return self._bar.update(n)

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        self._bar.set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)

    def close(self) -> None:
        close = getattr(self._bar, "close", None)
        if close is not None:
            close()

    def _description(self, desc: str | None) -> str | None:
        scope = _progress_scope.get()
        if desc is None or scope is None or scope.description_prefix is None:
            return desc
        return f"{scope.description_prefix} {desc}"

    def _position(self) -> int:
        scope = _progress_scope.get()
        if scope is not None and scope.worker_slot is not None:
            workers = scope.workers if scope.workers is not None else scope.worker_slot + 1
            return 1 + self._node.local_row * workers + scope.worker_slot
        return self._node.local_row

    def _leave(self, leave: bool | None) -> bool:
        if leave is not None:
            return leave
        scope = _progress_scope.get()
        if scope is not None and scope.leave is not None:
            return scope.leave
        return self._node.local_row == 0


class ProgressManager:
    def __init__(self, *, workers: int, context_name: str = "spawn") -> None:
        if workers < 1:
            raise ValueError("ProgressManager requires at least one worker.")
        self.workers = workers
        self.mp_context = mp.get_context(context_name)
        self._lock: Any | None = None
        self._log_queue: Any | None = None
        self._log_listener: logging.handlers.QueueListener | None = None
        self._scope_token: contextvars.Token[_ProgressScope | None] | None = None
        self._previous_tqdm_lock: Any | None = None

    def __enter__(self) -> ProgressManager:
        self._previous_tqdm_lock = tqdm.get_lock()
        self._lock = self.mp_context.RLock()
        self._log_queue = self.mp_context.Queue()
        self._log_listener = logging.handlers.QueueListener(
            self._log_queue,
            _make_tqdm_logging_handler(logging.INFO),
            respect_handler_level=True,
        )
        self._log_listener.start()
        tqdm.set_lock(self._lock)
        self._scope_token = _progress_scope.set(_ProgressScope(workers=self.workers))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._log_listener is not None:
            self._log_listener.stop()
            self._log_listener = None
        if self._scope_token is not None:
            _progress_scope.reset(self._scope_token)
            self._scope_token = None
        if self._previous_tqdm_lock is not None:
            tqdm.set_lock(self._previous_tqdm_lock)
            self._previous_tqdm_lock = None
        if self._log_queue is not None:
            with contextlib.suppress(Exception):
                self._log_queue.close()
            self._log_queue = None
        self._lock = None

    def worker_initargs(self) -> tuple[Any, int, Any]:
        if self._lock is None or self._log_queue is None:
            raise RuntimeError("ProgressManager must be entered before worker_initargs().")
        return self._lock, self.workers, self._log_queue

    @staticmethod
    def initialize_worker(tqdm_lock: Any, workers: int, log_queue: Any) -> None:
        tqdm.set_lock(tqdm_lock)
        _progress_scope.set(_ProgressScope(workers=workers))
        _configure_tqdm_queue_logging(log_queue)

    @staticmethod
    @contextlib.contextmanager
    def worker_scope(
        *,
        worker_slot: int,
        description_prefix: str | None = None,
        leave: bool | None = False,
        workers: int | None = None,
    ) -> Generator[None]:
        current_scope = _progress_scope.get()
        resolved_workers = (
            workers
            if workers is not None
            else (
                current_scope.workers
                if current_scope is not None and current_scope.workers is not None
                else worker_slot + 1
            )
        )
        if worker_slot < 0 or worker_slot >= resolved_workers:
            raise ValueError(
                f"worker_slot={worker_slot} is outside the configured workers={resolved_workers}."
            )
        # 每个任务从自己的进度树根开始；父进程总进度条固定占用第 0 行。
        scope_token = _progress_scope.set(
            _ProgressScope(
                workers=resolved_workers,
                worker_slot=worker_slot,
                description_prefix=description_prefix,
                leave=leave,
            )
        )
        node_token = _active_progress_node.set(None)
        try:
            yield
        finally:
            _active_progress_node.reset(node_token)
            _progress_scope.reset(scope_token)
