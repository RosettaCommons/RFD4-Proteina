"""One error body for every failure, with codes a caller can act on.

The code set and its status mapping follow the downstream RFD4-Proteina service,
so a client can handle both without a translation table. Two choices look wrong
and are not:

- ``service_busy`` is **503, not 429**. A full queue means this replica has no
  free worker, not that the caller is sending too fast; a load balancer should
  route elsewhere rather than ask the client to back off.
- ``out_of_memory`` is **413, a 4xx**. Retrying the same job exhausts the same
  GPU again. What the caller can change is the input or the replicate count.

``not_found`` and ``conflict`` are additions: this sample polls for a result,
so it has request-scoped routes the synchronous service has no equivalent for.
"""

from enum import StrEnum
from http import HTTPStatus
from typing import Any


class ErrorCode(StrEnum):
    SCHEMA_VIOLATION = "schema_violation"
    INVALID_REQUEST = "invalid_request"
    INVALID_INPUT = "invalid_input"
    REQUEST_TOO_LARGE = "request_too_large"
    OUT_OF_MEMORY = "out_of_memory"
    MODEL_UNAVAILABLE = "model_unavailable"
    SERVICE_BUSY = "service_busy"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INTERNAL_ERROR = "internal_error"


#: Total over :class:`ErrorCode`, so a handler never has to guess a status and
#: the two cannot drift. ``test_api.py`` pins the totality.
STATUS_FOR_CODE: dict[ErrorCode, int] = {
    ErrorCode.SCHEMA_VIOLATION: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.INVALID_REQUEST: HTTPStatus.BAD_REQUEST,
    ErrorCode.INVALID_INPUT: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.REQUEST_TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    ErrorCode.OUT_OF_MEMORY: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    ErrorCode.MODEL_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.SERVICE_BUSY: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorCode.CONFLICT: HTTPStatus.CONFLICT,
    ErrorCode.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}


class ApiError(Exception):
    """A failure the caller can act on, rendered as the declared error body."""

    def __init__(self, code: ErrorCode, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []

    @property
    def status(self) -> int:
        return int(STATUS_FOR_CODE[self.code])

    def body(self) -> dict[str, Any]:
        return {"error": {"code": str(self.code), "message": self.message, "details": self.details}}
