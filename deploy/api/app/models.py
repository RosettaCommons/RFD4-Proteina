"""The supported NIM request subset and this service's asynchronous responses."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ErrorBody

Task = Literal[
    "antibody_design",
    "enzyme_design",
    "generation",
    "inverse_folding",
    "motif_scaffolding",
    "partial_diffusion",
    "small_molecule_binder",
]
TASKS = get_args(Task)
UNSUPPORTED_TASKS = frozenset({"partial_diffusion", "inverse_folding"})
MAX_SEED = 2147483647


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputFile(StrictRequest):
    content: str = Field(min_length=1)
    filename: str = Field(default="input.cif", min_length=1, max_length=255, description="Label only; never a path.")

    @field_validator("content")
    @classmethod
    def utf8_content(cls, value: str) -> str:
        value.encode("utf-8")
        return value


class AnnotatedCifInput(StrictRequest):
    format: Literal["annotated_cif"]
    file: InputFile
    example_id: str = Field(default="design", pattern=r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}$")


class Sampling(StrictRequest):
    num_replicates: int = Field(default=1, ge=1, strict=True)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED, strict=True)


class DesignRequest(StrictRequest):
    task: Task
    input: AnnotatedCifInput
    sampling: Sampling = Field(default_factory=Sampling)


class JobSettings(BaseModel):
    num_replicates: int
    seed: int


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    task: Task | None = None
    example_id: str | None = None
    settings: JobSettings | None = None
    created_at: float | None = None
    updated_at: float | None = None
    started_at: float | None = None
    duration_seconds: float | None = None
    checkpoint_loads: int | None = None
    server_instance_id: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    error: ErrorBody | None = None


class Artifact(BaseModel):
    name: str
    download: str


class Results(BaseModel):
    id: str
    artifacts: list[Artifact]


class TaskSupport(BaseModel):
    supported: list[Task]
    unsupported: list[Task]


class Limits(BaseModel):
    max_request_bytes: int
    max_tokens: int
    max_replicates: int
    queue_depth: int


class Info(BaseModel):
    api_version: Literal["1.0.0"] = "1.0.0"
    tasks: TaskSupport
    limits: Limits
    submission: Literal["asynchronous"] = "asynchronous"


class Live(BaseModel):
    status: Literal["alive"]


class Ready(BaseModel):
    status: Literal["ready"]
    gpu: str
    checkpoint_loads: int
    server_instance_id: str
