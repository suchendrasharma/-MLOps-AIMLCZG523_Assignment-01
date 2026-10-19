"""Structured JSON logging: one JSON object per line, ready for kubectl logs / log shippers."""

import logging
import os
import sys

from pythonjsonlogger import jsonlogger


def json_formatter() -> logging.Formatter:
    return jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )


class _DropColorMessage(logging.Filter):
    """uvicorn attaches an ANSI-coloured copy of each message; it is noise in JSON logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.pop("color_message", None)
        return True


def configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(json_formatter())
    handler.addFilter(_DropColorMessage())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    # Our middleware writes one structured line per request; silence the plain-text duplicate.
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).propagate = False
    return logging.getLogger("heart.api")
