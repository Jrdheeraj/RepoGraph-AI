import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.repositories.repository_repository import RepositoryRepository
from backend.schemas.repository import RepositoryResponse
from backend.services.github_service import (
    GitHubService,
    InvalidGitHubUrlError,
    RepositoryAlreadyExistsError,
    RepositoryCloneError,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/repos", tags=["repositories"])


class RepositoryImportRequest(BaseModel):
    github_url: str = Field(..., min_length=1, max_length=500)


def get_repository_repository(db: Session = Depends(get_db)) -> RepositoryRepository:
    return RepositoryRepository(db)


def get_github_service(
    repository_repository: RepositoryRepository = Depends(get_repository_repository),
) -> GitHubService:
    return GitHubService(repository_repository)


@router.post(
    "/import",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
)
def import_repository(
    payload: RepositoryImportRequest,
    github_service: GitHubService = Depends(get_github_service),
) -> RepositoryResponse:
    try:
        repository = github_service.import_repository(payload.github_url)
        return RepositoryResponse.model_validate(repository)
    except InvalidGitHubUrlError as exc:
        logger.warning(
            "repository_import_invalid_url",
            extra={"github_url": payload.github_url},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RepositoryAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except RepositoryCloneError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while importing repository.",
        ) from exc


@router.get("", response_model=list[RepositoryResponse])
def list_repositories(
    repository_repository: RepositoryRepository = Depends(get_repository_repository),
) -> list[RepositoryResponse]:
    try:
        repositories = repository_repository.list()
        return [RepositoryResponse.model_validate(repository) for repository in repositories]
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while listing repositories.",
        ) from exc


@router.get("/{repo_id}", response_model=RepositoryResponse)
def get_repository(
    repo_id: UUID,
    repository_repository: RepositoryRepository = Depends(get_repository_repository),
) -> RepositoryResponse:
    try:
        repository = repository_repository.get_by_id(repo_id)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching repository.",
        ) from exc

    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        )

    return RepositoryResponse.model_validate(repository)
