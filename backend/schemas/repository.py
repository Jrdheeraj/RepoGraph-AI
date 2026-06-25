from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RepositoryCreate(BaseModel):
    github_url: str = Field(..., min_length=1, max_length=500)
    name: str = Field(..., min_length=1, max_length=255)
    owner: str = Field(..., min_length=1, max_length=255)
    language: str | None = Field(default=None, max_length=100)
    status: str = Field(default="pending", min_length=1, max_length=50)


class RepositoryResponse(BaseModel):
    id: UUID
    github_url: str
    name: str
    owner: str
    language: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
