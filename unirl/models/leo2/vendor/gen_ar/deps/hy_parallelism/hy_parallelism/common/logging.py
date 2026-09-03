"""
Logging utility functions.
"""
import sys
import os
import termios
import tty
import re
import select
import datetime
import logging
import threading
import time
import atexit
import signal
import multiprocessing
from typing import Optional, ParamSpec

import loguru
import loguru._defaults
from torch import distributed as dist

from hy_parallelism.parallel_states import get_parallel_state, is_parallel_state_initialized
from hy_parallelism.utils import get_taiji_user



# Global variable to track terminal settings that need to be restored
_terminal_settings_to_restore = None
_terminal_fd = None
_color_refresh_thread_started = False
_color_refresh_lock = threading.Lock()
_terminal_osc_thread_lock = threading.Lock()
_terminal_osc_process_lock = multiprocessing.Lock()




_colorize_loguru = True if get_taiji_user() == 'kevinkhwu' else loguru._defaults.LOGURU_COLORIZE

def _restore_terminal_settings():
    """Restore terminal settings if they were modified."""
    global _terminal_settings_to_restore, _terminal_fd
    if _terminal_settings_to_restore is not None and _terminal_fd is not None:
        try:
            termios.tcsetattr(_terminal_fd, termios.TCSANOW, _terminal_settings_to_restore)
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            # Last resort: try to reset terminal using stty sane
            raise
            try:
                import subprocess
                subprocess.run(['stty', 'sane'], check=False, timeout=1, 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        finally:
            _terminal_settings_to_restore = None
            _terminal_fd = None


# atexit.register(_restore_terminal_settings)

# def _signal_handler(signum, frame):
#     _restore_terminal_settings()
#     # Re-raise the signal with default handler
#     signal.signal(signum, signal.SIG_DFL)
#     os.kill(os.getpid(), signum)

# Register 这些会导致退出是卡死
# for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
#     try:
#         signal.signal(sig, _signal_handler)
#     except (ValueError, OSError):
#         # Signal not available on this platform
#         pass


def parse_rgb(resp: str) -> tuple[int, int, int] | None:
    match = re.search(r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", resp)
    if match:
        # TODO: At some point figure out why ghostty return RR
        # -> Seems to be the spec
        rgb = "".join(m[:2] for m in match.groups())
        # rgb is like #ffffff
        # convert hex string to (r, g, b) tuple
        assert len(rgb) == 6
        if len(rgb) == 6:
            r = int(rgb[0:2], 16)
            g = int(rgb[2:4], 16)
            b = int(rgb[4:6], 16)
            return (r, g, b)
        else:
            return "#" + rgb
    else:
        return None


def query_palette_color(index: int) -> Optional[str]:
    """Send OSC 4 query and parse reply."""
    seq = f"\033]4;{index};?\a"
    response = send_osc_query(seq)
    return parse_rgb(response)


def query_background_color():
    seq = "\033]11;?\a"
    response = send_osc_query(seq)
    return parse_rgb(response)


def query_foreground_color():
    seq = "\033]10;?\a"
    response = send_osc_query(seq)
    return parse_rgb(response)

def lock_decorator(lock: threading.Lock):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator

@lock_decorator(_terminal_osc_process_lock)
@lock_decorator(_terminal_osc_thread_lock)
def send_osc_query(seq) -> str:
    """
    Send an OSC query. To avoid locking up when something
    unexpected happens, the terminal has 1 second to respond
    and cannot write more than 1024 characters.
    If there is any issue, an empty string is returned.
    """
    response = ""

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
        global _terminal_settings_to_restore, _terminal_fd
        _terminal_settings_to_restore = old_settings
        _terminal_fd = fd
    except Exception as e:
        loguru.logger.warning(f'Fail to get terminal settings. {e}')
        return ""
    try:
        if not sys.stdout.isatty():
            raise Exception('stdout is not a tty')

        tty.setcbreak(fd)

        # tty.setraw(fd)
        sys.stdout.write(seq)
        sys.stdout.flush()
        timeout_s = 1
        # I am using a timeout when reading from `stdin`
        # to ensure that this won't block forever if the terminal
        # behaves oddly. This may only work on `UNIX` but
        # that should be fine for now.
        ready, _, _ = select.select([fd], [], [], timeout_s)
        last_char_was_esc = False
        if ready:
            for _ in range(1024):
                c = sys.stdin.read(1)
                if c == "\a" or (c == "\x5c" and last_char_was_esc):
                    break

                if c == "\x1b":
                    last_char_was_esc = True
                    continue
                else:
                    last_char_was_esc = False
                response += c
            else:
                response = ""
    except Exception as e:
        loguru.logger.warning(f'Fail to send OSC query: {e}')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    if len(response) == 0:
        print(f"[Terminal info]: No response from terminal. This may happen if tmux version is too old (e.g., does not support 24bit color query).")
    return response



def is_dark_terminal_bg() -> bool:
    def cloudclli_preference() -> bool:
        hour = datetime.datetime.now().hour
        return hour >= 18 or hour < 6
    user = get_taiji_user()
    print(f"[Terminal info]: User: {user}")

    # world_size = os.environ.get('WORLD_SIZE', '1')
    # if world_size == '1':
    #     rgb = query_background_color()
    # else:
    #     if dist.is_initialized():
    #         import torch
    #         if os.environ.get('RANK', '0') == '0':
    #             rgb = query_background_color()
    #         else:
    #             rgb = 0
    #         from hy_parallelism.utils import auto_broadcast
    #         rgb = auto_broadcast(rgb, src=0)
    #     else:
    #         return False
    rgb = query_background_color()

    if rgb is not None:
        r, g, b = rgb
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        print(f'[Terminal info]: Background rgb: {rgb}, brightness: {brightness}')
        return brightness < 128
    else:
        import subprocess
        try:
            if os.path.exists('/usr/bin/tmux'):
                tmux_version = subprocess.check_output(['/usr/bin/tmux', '-V'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
                print(f"[Terminal info]: /usr/bin/tmux version: {tmux_version}")
        except:
            ...
        try:
            if os.path.exists('/usr/local/bin/tmux'):
                tmux_version = subprocess.check_output(['/usr/local/bin/tmux', '-V'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
                print(f"[Terminal info]: /usr/local/bin/tmux version: {tmux_version}")
        except:
            ...

        user_preference = {
            'cloudclli': cloudclli_preference(),
            'kevinkhwu': True,
        }
        if user in user_preference:
            loguru.logger.info(f"[Terminal info]: User {user} preference: {user_preference[user]}")
            return user_preference[user]
        else:
            print('[Terminal info]: Falling back to default logging color scheme.')
            return False
    


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def flush_right(left_part: str, right_part: str, min_padding: int = 2, terminal_width: Optional[int] = None) -> str:
    """
    Flush the right_part to the rightmost position of the terminal, ensuring left_part is never overwritten.
    
    Args:
        left_part: The left part of the line (may contain ANSI color codes)
        right_part: The right part to flush to the right (may contain ANSI color codes)
        min_padding: Minimum padding between left and right parts when they fit on one line
        terminal_width: Terminal width, if None will be detected automatically
    
    Returns:
        Formatted string with right_part flushed to the right, with newline at the end
    """
    def get_terminal_size() -> tuple[int, int]:
        """
        健壮的获取终端尺寸函数，兼容所有重定向/管道场景
        :param default_cols: 无法获取时的默认列数
        :param default_rows: 无法获取时的默认行数
        :return: (列数, 行数)
        """
        # 优先级：stderr > stdin > stdout （按被重定向概率从低到高排序）
        fds_to_try = [sys.stderr.fileno(), sys.stdin.fileno(), sys.stdout.fileno()]
        # fds_to_try = [sys.stdout.fileno()]
        for fd in fds_to_try:
            try:
                return os.get_terminal_size(fd)
            except (OSError, ValueError):
                # OSError: fd不是终端/不支持ioctl；ValueError: fd无效
                continue
        # 所有fd都获取失败，返回默认值
        return (-1, -1)
    import re
    
    if terminal_width is None:
        try:
            # terminal_width = shutil.get_terminal_size().columns # could fail with pipelining. e.g. `python3 ./foo.py | tee`
            terminal_width = get_terminal_size()[0]
        except:
            terminal_width = 120
            raise
    
    # Calculate actual display length (without ANSI codes)
    left_plain = re.sub(r'<[^>]+>', '', left_part)
    right_plain = re.sub(r'<[^>]+>', '', right_part)
    
    # Check if everything fits on one line
    total_width_needed = len(left_plain) + min_padding + len(right_plain)
    
    if terminal_width == -1:
        return f"{left_part}  \t{right_part}\n"
    elif total_width_needed <= terminal_width:
        # Use ANSI escape code to move cursor to right position
        target_column = terminal_width - len(right_plain) + 1
        return left_part + f"\033[{target_column}G" + right_part + "\n"
    else:
        # Terminal too narrow - put right_part on a new line, right-aligned
        target_column = terminal_width - len(right_plain) + 1
        return left_part + "\n" + (" " * (target_column - 1)) + right_part + "\n"


try:
    from loguru import logger as loguru_logger

    _has_loguru = True
except ImportError:
    _has_loguru = False

__LOGGER_COLOR_SETUP = False
def setup_logger_color(force=False):
    global __LOGGER_COLOR_SETUP
    if __LOGGER_COLOR_SETUP and not force:
        return

    # 等 initialized 之后再做，因为获取颜色需要通信
    # if os.environ.get('WORLD_SIZE', '1') != '1' and not dist.is_initialized():
    #     return

    # 如果 HY_PARALLELISM_DEFAULT_LOGGING_COLOR=1 的时候，是绝对不能替换颜色的
    if os.environ.get('HY_PARALLELISM_DEFAULT_LOGGING_COLOR', '0') == '0' and get_taiji_user() == 'kevinkhwu':
        # if is_dark_terminal_bg():
        if True:
            # 深色背景配色
            loguru_logger.level('DEBUG', color="<fg 100,100,100>")
            loguru_logger.level('INFO', color="<white>")
            loguru_logger.level('WARNING', color="<yellow>")
            loguru_logger.level('ERROR', color="<red>")
            # loguru_logger.level('CRITICAL', color="<fg #00ffff><bg #ff0000>")
            loguru_logger.level('CRITICAL', color="<white><bg #ff0000>")
        else:
            # 浅色背景（白色）配色 - 使用深色文字以确保可读性
            loguru_logger.level('DEBUG', color="<fg 80,80,80>")  # 深灰色
            loguru_logger.level('INFO', color="<fg #000080>")  # 深蓝色
            loguru_logger.level('WARNING', color="<fg #bb6600>")  # 深橙色
            loguru_logger.level('ERROR', color="<fg #cc0000>")  # 深红色
            loguru_logger.level('CRITICAL', color="<fg #cc0000><bg #ffff00>")  # 深红色文字配黄色背景

    __LOGGER_COLOR_SETUP = True


def _periodic_color_refresh():
    while True:
        try:
            setup_logger_color(force=True)
        except Exception as e:
            loguru_logger.warning(f"Failed to refresh logger color: {e}")
            raise
        time.sleep(60 * 60)  # 60 mins


def start_periodic_color_refresh():
    global _color_refresh_thread_started
    with _color_refresh_lock:
        if not _color_refresh_thread_started:
            thread = threading.Thread(target=_periodic_color_refresh, daemon=True)
            thread.start()
            _color_refresh_thread_started = True

if _has_loguru:
    _loguru_configured = False

    class LoguruLoggerWrapper:
        """
        A wrapper around loguru's logger to provide a similar interface to the standard logging.Logger,
        including the custom `rank0` parameter in `info`.
        """

        def __init__(self, name: Optional[str] = None):
            logger = loguru_logger

            if name:
                # Patch to include the logger name in the record
                logger = logger.patch(lambda record: record.update(name=name))

            self._logger = logger

        def info(self, msg, *args, rank0=False, **kwargs):
            logger = self._logger.opt(depth=1)
            if rank0:
                if get_global_rank() == 0:
                    logger.info(msg, *args, **kwargs)
            else:
                logger.info(msg, *args, **kwargs)

        def setLevel(self, level):
            # loguru's level is configured per-sink, this is for API compatibility
            pass

        def addHandler(self, handler):
            # loguru's handlers (sinks) are managed globally, this is for API compatibility
            pass

        def __getattr__(self, name: str):
            """Forward other calls to the underlying loguru logger."""
            return getattr(self._logger, name)


class RankAwareLogger(logging.Logger):
    def info(self, msg, *args, rank0=False, **kwargs):
        if rank0:
            if get_global_rank() == 0:
                super().info(msg, *args, **kwargs)
        else:
            super().info(msg, *args, **kwargs)


logging.setLoggerClass(RankAwareLogger)

_default_handler = logging.StreamHandler(sys.stdout)
_default_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s "
        + (f"[Rank:{get_global_rank()}]" if get_world_size() > 1 else "")
        + (f"[LocalRank:{get_local_rank()}]" if get_world_size() > 1 else "")
        + "[%(threadName).12s][%(name)s][%(levelname).5s] "
        + "%(message)s"
    )
)

logger_level_mapping = dict() # name -> level


def get_logger(
    name: Optional[str] = None, 
    logger_level_mapping: dict = dict(),
) -> logging.Logger:
    """
    Get a logger.
    It will use loguru if available, otherwise it will fallback to standard logging.
    """
    if _has_loguru:
        global _loguru_configured
        if logger_level_mapping:
            _loguru_configured = False
        if not _loguru_configured:
            loguru_logger.remove()  # remove default handler

            loguru_logger.configure(
                patcher=lambda record: record["extra"].update(
                    global_rank=get_global_rank(), local_rank=get_local_rank(), node_id=get_global_rank() // 8, world_size=get_world_size()
                )
            )

            level_score = {
                'DEBUG': 0,
                'INFO': 1,
                'WARN': 2,
                'WARNING': 2,
                'ERROR': 3,
                'CRITICAL': 4,
            }

            def filter_func(record):
                name = record["name"]
                current_level = record['level'].name
                for logger_name, level in logger_level_mapping.items():
                    if name.startswith(logger_name):
                        return level_score[current_level] - level_score[level] >= 0
                return True

            def format_func(record):
                # Build the left part of the log line
                time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                rank_str = ""
                if get_world_size() > 1:
                    # rank_str = f"[Rank:n{record['extra']['node_id']}-l{record['extra']['local_rank']} ({record['extra']['global_rank']}/{record['extra']['world_size']})]"
                    extra_ranks = ''
                    if is_parallel_state_initialized():
                        parallel_state = get_parallel_state()
                        if parallel_state.pp > 1:
                            extra_ranks += f"-pp{parallel_state.pp_rank}[{parallel_state.pp}]"
                        if parallel_state.dp_size > 1:
                            extra_ranks += f"-dp{parallel_state.dp_rank}[{parallel_state.dp_size}]"
                        if parallel_state.cp > 1:
                            extra_ranks += f"-cp{parallel_state.cp_rank}[{parallel_state.cp}]"
                        if parallel_state.ep > 1:
                            extra_ranks += f"-ep{parallel_state.ep_rank}[{parallel_state.ep}]"
                        if parallel_state.tp > 1:
                            extra_ranks += f"-tp{parallel_state.tp_rank}[{parallel_state.tp}]"
                        if parallel_state.etp > 1:
                            extra_ranks += f"-etp{parallel_state.etp_rank}[{parallel_state.etp}]"

                        from hy_parallelism.parallel_states import _PARALLEL_STATE_KEY, _DEFAULT_PARALLEL_STATE_KEY
                        if _PARALLEL_STATE_KEY != _DEFAULT_PARALLEL_STATE_KEY:
                            extra_ranks += f"-tag:{_PARALLEL_STATE_KEY}"
                    
                    rank_str = f"[R{record['extra']['global_rank']}[{record['extra']['world_size']}]-n{record['extra']['node_id']}g{record['extra']['local_rank']}{extra_ranks}]"
                level_str = f"[{record['level'].name:.5s}]"
                message_str = record['message']
                
                # Build left part (timestamp + rank + level + message)
                left_part = f"{time_str} {rank_str}{level_str} {message_str}"
                left_part_format = f"<green>{time_str}</green> {rank_str}{level_str} <level><bold>{{message}}</bold></level>"
                
                # Build right part (file:line info)
                # right_part = f"({record['name']}:<cyan>{record['file']}:{record['line']}</cyan>)"
                right_part = f"({record['name']}:{record['file']}:{record['line']})"
                right_part_format = f"({{name}}:<cyan>{{file}}:{{line}}</cyan>)"
                

                def get_terminal_size() -> tuple[int, int]:
                    # 优先级：stderr > stdin > stdout （按被重定向概率从低到高排序）
                    # 工蜂 CI test 可能会在 fileno 抛异常
                    try:
                        fds_to_try = [sys.stderr.fileno(), sys.stdin.fileno(), sys.stdout.fileno()]
                        # fds_to_try = [sys.stdout.fileno()]
                        for fd in fds_to_try:
                            try:
                                return os.get_terminal_size(fd)
                            except (OSError, ValueError):
                                # OSError: fd不是终端/不支持ioctl；ValueError: fd无效
                                continue
                    except:
                        pass
                    # 所有fd都获取失败，返回默认值
                    return (-1, -1)
                import re
                
                if os.environ.get('HY_PARALLELISM_LOGGING_DISABLE_FILENAME_FLUSH', '0') == '1':
                    terminal_width = -1
                else:
                    try:
                        # terminal_width = shutil.get_terminal_size().columns # could fail with pipelining. e.g. `python3 ./foo.py | tee`
                        terminal_width = get_terminal_size()[0]
                    except:
                        terminal_width = 120
                        raise
                
                # Remove thins like <level> <bold>
                # left_plain = re.sub(r'<[^>]+>', '', left_part) 
                # right_plain = re.sub(r'<[^>]+>', '', right_part)

                left_plain = left_part
                right_plain = right_part

                # 如果 message 里面有 <xxx> ，在上面被替换掉，会导致长度计算不准确，这里补回来
                # missed_len = len(message_str) - len(re.sub(r'<[^>]+>', '', message_str))
                
                # Check if everything fits on one line
                total_width_needed = len(left_plain) + len(right_plain) + 2 # + missed_len
                
                if terminal_width == -1:
                    # 无 TTY 时也必须返回带占位符的模板，不能把 message 直接拼进去，
                    # 否则 Loguru 的 format_map 会把消息里的 {} 当成占位符解析 → KeyError
                    return left_part_format + "  \t" + right_part_format + "\n"
                    # return f"{left_part}  \t{right_part}\n"
                elif total_width_needed <= terminal_width:
                    # Use ANSI escape code to move cursor to right position
                    target_column = terminal_width - len(right_plain) + 1
                    return left_part_format + f"\033[{target_column}G" + right_part_format + "\n"
                else:
                    # Terminal too narrow - put right_part on a new line, right-aligned
                    target_column = terminal_width - len(right_plain) + 1
                    return left_part_format + "\n" + (" " * (target_column - 1)) + right_part_format + "\n"
                    # return left_part_escaped + " \t" + right_part_escaped + "\n"

            loguru_logger.add(
                sys.stdout,
                format=format_func,
                # level="INFO",
                filter=filter_func,
                colorize=_colorize_loguru,
            )

            setup_logger_color()
            # if get_taiji_user() == 'cloudclli':
            #     start_periodic_color_refresh()  # 黑白切换，需要定期更新
            _loguru_configured = True
        return LoguruLoggerWrapper(name)  # type: ignore

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(_default_handler)
    logger.setLevel(logging.INFO)
    return logger

def mute_loggers(mute_logger_names: list[str], level: str = 'WARNING'):
    get_logger(
        logger_level_mapping={
            mute_logger_name: level for mute_logger_name in mute_logger_names
        }
    )

EXISTING_LOGGER_LEVEL_MAPPING = dict()
DEBUG_LOGGER_NAME = 'hy_parallelism.debug'

def configure_logger_level(logger_level_mapping: dict):
    global EXISTING_LOGGER_LEVEL_MAPPING
    get_logger(
        logger_level_mapping=logger_level_mapping
    )
    EXISTING_LOGGER_LEVEL_MAPPING = logger_level_mapping.copy()

def update_logger_level(logger_level_mapping):
    EXISTING_LOGGER_LEVEL_MAPPING.update(logger_level_mapping)
    configure_logger_level(EXISTING_LOGGER_LEVEL_MAPPING)

def get_debug_logger():
    return get_logger(DEBUG_LOGGER_NAME)

def debug_log(msg):
    if os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
        get_debug_logger().opt(depth=1).debug(msg)

def rank0_log(msg, level='INFO', *args, **kwargs):
    if get_global_rank() == 0:
        loguru_logger.opt(depth=1).log(level, msg, *args, **kwargs)


_trace_log_path: str | None = None


def set_trace_log_path(path: str | None) -> None:
    global _trace_log_path
    _trace_log_path = path


def trace_log(msg: str) -> None:
    if _trace_log_path is None:
        loguru_logger.info(msg)
        return
    try:
        from pathlib import Path
        Path(_trace_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(_trace_log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass


if os.environ.get('ENABLE_HY_PARALLELISM_LOGGING', '0') == '1':
    get_logger(__name__)
    get_debug_logger()

    if os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
        configure_logger_level({
            DEBUG_LOGGER_NAME: 'DEBUG',
        })
    else:
        configure_logger_level({
            DEBUG_LOGGER_NAME: 'ERROR', # Use ERROR level to avoid spamming the log
        })
