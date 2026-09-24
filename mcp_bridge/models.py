"""Subset of the existing API contract exposed to MCP tools."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkoutInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: date
    exercise: str = Field(min_length=1, max_length=200)
    set_number: int = Field(ge=1)
    reps: int | None = Field(default=None, ge=0)
    value: float | None = None
    unit: str | None = None
    cycle: int | None = Field(default=None, ge=0)
    week: int | None = Field(default=None, ge=0)
    iso_week: int | None = Field(default=None, ge=1, le=53)
    day: int | None = Field(default=None, ge=0)
    notes: str | None = None
    tags: str | None = None

    @field_validator("exercise")
    @classmethod
    def nonempty_exercise(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("exercise must not be blank")
        return value.strip()


class MetconInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: date
    name: str = Field(min_length=1, max_length=200)
    workout_type: Literal["for_time", "amrap", "emom", "chipper", "interval", "other"]
    description: str | None = None
    score_time_seconds: int | None = Field(default=None, ge=0)
    score_rounds: int | None = Field(default=None, ge=0)
    score_reps: int | None = Field(default=None, ge=0)
    score_display: str | None = None
    rx: Literal["rx", "scaled", "rx_plus"] | None = None
    time_cap_seconds: int | None = Field(default=None, ge=0)
    cycle: int | None = Field(default=None, ge=0)
    week: int | None = Field(default=None, ge=0)
    iso_week: int | None = Field(default=None, ge=1, le=53)
    day: int | None = Field(default=None, ge=0)
    notes: str | None = None
    tags: str | None = None

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value.strip()
