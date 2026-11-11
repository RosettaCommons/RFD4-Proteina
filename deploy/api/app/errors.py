"""NIM error vocabulary, plus not_found/conflict for asynchronous jobs.

A full queue is service_busy (503); GPU OOM asks the caller to shrink the job (413).
"""

from enum import StrEnum
from http import HTTPStatus
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
        self.code = code
        self.message = message
        self.details = details or []

    @property
    def status(self) -> int:
        return int(STATUS_FOR_CODE[self.code])

    def body(self) -> dict[str, Any]:
        return {"error": {"code": str(self.code), "message": self.message, "details": self.details}}


def error_body(exc: Exception) -> dict[str, Any]:
    """Preserve actionable failures; unexpected details belong in server logs."""
    if isinstance(exc, ApiError):
        return exc.body()["error"]
    return ApiError(ErrorCode.INTERNAL_ERROR, "Internal error.").body()["error"]
