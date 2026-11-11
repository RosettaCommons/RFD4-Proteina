"""NIM error vocabulary, plus not_found/conflict for asynchronous jobs.

A full queue is service_busy (503); GPU OOM asks the caller to shrink the job (413).
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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


STATUS_FOR_CODE: dict[ErrorCode, int] = {
    ErrorCode.SCHEMA_VIOLATION: 422,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.REQUEST_TOO_LARGE: 413,
    ErrorCode.OUT_OF_MEMORY: 413,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.SERVICE_BUSY: 503,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    """A failure the caller can act on, rendered as the declared error body."""

    def __init__(self, code: ErrorCode, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.status = STATUS_FOR_CODE[code]
        self.error = {"code": str(code), "message": message, "details": details or []}

    def body(self) -> dict[str, Any]:
        return {"error": self.error}


def error_body(exc: Exception) -> dict[str, Any]:
    """Preserve actionable failures; unexpected details belong in server logs."""
    if isinstance(exc, ApiError):
        return exc.body()["error"]
    return ApiError(ErrorCode.INTERNAL_ERROR, "Internal error.").body()["error"]
