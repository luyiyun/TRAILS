"""兼容嵌套、多进程进度条的 tqdm 日志与位置协调工具。"""

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
    """保存当前进程或 worker 的进度条布局上下文。"""

    workers: int | None = None
    worker_slot: int | None = None
    description_prefix: str | None = None
    leave: bool | None = None


@dataclass(frozen=True)
class _ProgressNode:
    """记录嵌套进度条在当前进度树中的本地行号。"""

    parent: _ProgressNode | None
    local_row: int

    @classmethod
    def from_active_parent(cls) -> _ProgressNode:
        """根据当前活动父节点创建下一层进度节点。"""
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
    """把 worker 描述前缀注入日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """附加当前进度作用域前缀并保留日志。"""
        scope = _progress_scope.get()
        if scope is not None and scope.description_prefix is not None:
            record.progress_description_prefix = scope.description_prefix
        return True


class CompactTqdmFormatter(logging.Formatter):
    """将日志压缩为适合与 tqdm 同屏显示的单行文本。"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化、添加 worker 前缀并按终端宽度截短日志。"""
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
    """通过 :func:`tqdm.write` 输出且不破坏活动进度条的日志处理器。"""

    def emit(self, record: logging.LogRecord) -> None:
        """安全格式化并写出一条日志记录。"""
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _shorten_log_line(message: str) -> str:
    """折叠空白并把日志截短到当前终端宽度。"""
    compact_message = " ".join(message.split())
    return textwrap.shorten(
        compact_message,
        width=_terminal_log_width(),
        placeholder=" ...",
    )


def _terminal_log_width() -> int:
    """返回至少 40 列的可用单行日志宽度。"""
    return max(40, shutil.get_terminal_size(fallback=(120, 20)).columns - 1)


def _make_tqdm_logging_handler(level: int) -> TqdmLoggingHandler:
    """创建带紧凑格式和进度前缀过滤器的日志处理器。"""
    handler = TqdmLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(CompactTqdmFormatter("%(levelname)s:%(name)s:%(message)s"))
    _ensure_progress_log_filter(handler)
    return handler


def _ensure_progress_log_filter(handler: logging.Handler) -> None:
    """确保日志处理器只安装一个进度作用域过滤器。"""
    if not any(isinstance(log_filter, _ProgressLogFilter) for log_filter in handler.filters):
        handler.addFilter(_ProgressLogFilter())


def configure_tqdm_logging(level: int = logging.INFO) -> None:
    """配置根日志器，使终端日志与 tqdm 进度条兼容。

    已存在 TRAILS 处理器时更新其级别和格式；否则移除普通终端流处理器并安装
    :class:`TqdmLoggingHandler`。文件处理器保持不变，同时捕获 Python warning。

    参数：
        level: 根日志器和 tqdm 处理器使用的日志级别。
    """
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
    """在 worker 中把非文件日志转发到父进程队列。"""
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
    """捕获 warning，并过滤已知不影响结果的高噪声提示。"""
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
    """自动协调嵌套层级和多 worker 行位置的 tqdm 包装器。

    可以像普通迭代器或上下文管理器一样使用。进度条会继承当前 worker 的描述
    前缀和 ``leave`` 策略；嵌套进度条依据进度树深度分配独立终端行。
    """

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        *,
        desc: str | None = None,
        leave: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """创建进度条，并解析作用域控制的描述、位置和保留策略。"""
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
        """迭代底层对象，并在结束时恢复父节点和关闭进度条。"""
        token = _active_progress_node.set(self._node)
        try:
            yield from self._bar
        finally:
            _active_progress_node.reset(token)
            self.close()

    def __enter__(self) -> ProgressBar[T]:
        """进入当前进度节点上下文并返回自身。"""
        self._enter_token = _active_progress_node.set(self._node)
        enter = getattr(self._bar, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """退出底层进度条并恢复先前活动节点。"""
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
        """把未定义属性代理到底层 tqdm 实例。"""
        return getattr(self._bar, name)

    def update(self, n: int | float = 1) -> bool | None:
        """将进度增加指定数量并返回 tqdm 刷新结果。"""
        return self._bar.update(n)

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        """更新进度条尾部显示的指标。"""
        self._bar.set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)

    def close(self) -> None:
        """关闭底层进度条。"""
        close = getattr(self._bar, "close", None)
        if close is not None:
            close()

    def _description(self, desc: str | None) -> str | None:
        """在描述前添加当前 worker 前缀。"""
        scope = _progress_scope.get()
        if desc is None or scope is None or scope.description_prefix is None:
            return desc
        return f"{scope.description_prefix} {desc}"

    def _position(self) -> int:
        """根据嵌套深度、worker 数和槽位计算终端行号。"""
        scope = _progress_scope.get()
        if scope is not None and scope.worker_slot is not None:
            workers = scope.workers if scope.workers is not None else scope.worker_slot + 1
            return 1 + self._node.local_row * workers + scope.worker_slot
        return self._node.local_row

    def _leave(self, leave: bool | None) -> bool:
        """解析显式、作用域或默认的进度条保留策略。"""
        if leave is not None:
            return leave
        scope = _progress_scope.get()
        if scope is not None and scope.leave is not None:
            return scope.leave
        return self._node.local_row == 0


class ProgressManager:
    """协调父进程日志、共享 tqdm 锁和多 worker 进度布局。

     上下文进入时创建指定 multiprocessing 上下文的锁与日志队列，在父进程启动
    队列监听器，并为 worker 进度条保留布局信息；退出时恢复原 tqdm 锁并释放
     所有资源。
    """

    def __init__(self, *, workers: int, context_name: str = "spawn") -> None:
        """配置正 worker 数量和 multiprocessing 启动上下文。"""
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
        """启动共享锁、日志队列和父进程监听器。"""
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
        """停止监听器、恢复上下文和锁，并关闭日志队列。"""
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
        """返回 worker 初始化所需的共享锁、worker 数和日志队列。

        异常：
            RuntimeError: 当管理器尚未进入上下文时抛出。
        """
        if self._lock is None or self._log_queue is None:
            raise RuntimeError("ProgressManager must be entered before worker_initargs().")
        return self._lock, self.workers, self._log_queue

    @staticmethod
    def initialize_worker(tqdm_lock: Any, workers: int, log_queue: Any) -> None:
        """在子进程中安装共享 tqdm 锁、布局作用域和队列日志。"""
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
        """为一个 worker 任务设置临时进度布局和描述前缀。

        参数：
            worker_slot: 当前任务占用的零基 worker 槽位。
            description_prefix: 添加到进度条和日志前的可选文本。
            leave: worker 进度条结束后是否保留。
            workers: 可选的显式总 worker 数，默认继承当前作用域。

        异常：
            ValueError: 当槽位不在 ``[0, workers)`` 内时抛出。
        """
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
