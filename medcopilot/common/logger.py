import logging
import re
import sys
from pathlib import Path

from loguru import logger

from medcopilot.common.config import config

_SECRET_KEY_NAME = (
    r"(?:[A-Za-z0-9_.-]*[_-])?"
    r"(?:access[_-]?token|client[_-]?secret|client[_-]?id|api[_-]?key|apikey|signing[_-]?key|qianfan[_-]?(?:sk|ak)|secret|token|key|password)"
)
_SECRET = re.compile(
    rf"(?i)(\b{_SECRET_KEY_NAME}\s*=\s*)[^&\s\"']+"
    r"|(\b(?:Bearer|Basic)\s+)[A-Za-z0-9._~+\-/=]+"
    rf"|([\"']{_SECRET_KEY_NAME}[\"']\s*:\s*[\"'])[^\"']+"
)
_EXTRA_SECRET_RE = re.compile(rf"(?i)^{_SECRET_KEY_NAME}$|^authorization$")


def redact_log(text: str) -> str:
    return _SECRET.sub(lambda match: (match.group(1) or match.group(2) or match.group(3)) + "***", text)


def _patch_record(record):
    record["message"] = redact_log(record["message"])
    record["extra"].setdefault("request_id", None)
    for key in list(record["extra"]):
        if _EXTRA_SECRET_RE.match(str(key)):
            record["extra"][key] = "***"


class QuerySecretFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact_log(record.getMessage())
        record.args = ()
        return True


for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).addFilter(QuerySecretFilter())

logger.remove()
logger.configure(patcher=_patch_record)
_json = config.log_format == "json"
logger.add(
    sys.stdout,
    colorize=not _json,
    serialize=_json,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=config.log_level,
    diagnose=False,
    backtrace=False,
)

# File sink is opt-out and never fatal at import in read-only deployments, and a
# malformed LOG_FILE must not be fatal either: loguru formats the path with
# str.format_map, so a value containing a brace field (LOG_FILE="/var/log/${HOME}-x")
# raises KeyError while the vault of this import is an app that cannot start. A
# bad log path degrades file logging and says so.
_log_file = (config.log_file or "").strip()
if _log_file:
    try:
        Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            _log_file,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            serialize=_json,
            level=config.log_level,
            encoding="utf-8",
            diagnose=False,
            backtrace=False,
        )
    except (OSError, ValueError, KeyError) as _exc:
        logger.warning("File logging disabled for '{}': {}", _log_file, _exc)

__all__ = ["logger", "redact_log"]
