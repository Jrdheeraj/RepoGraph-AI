from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.models.repository import Repository


class RepositoryRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        repository_id: UUID,
        github_url: str,
        name: str,
        owner: str,
        language: str | None,
        status: str,
    ) -> Repository:
        repository = Repository(
            id=repository_id,
            github_url=github_url,
            name=name,
            owner=owner,
            language=language,
            status=status,
        )

        try:
            self._db.add(repository)
            self._db.commit()
            self._db.refresh(repository)
        except SQLAlchemyError:
            self._db.rollback()
            raise

        return repository

    def get_by_id(self, repository_id: UUID) -> Repository | None:
        return self._db.get(Repository, repository_id)

    def get_by_github_url(self, github_url: str) -> Repository | None:
        statement = select(Repository).where(Repository.github_url == github_url)
        return self._db.scalars(statement).first()

    def list(self) -> list[Repository]:
        statement = select(Repository).order_by(Repository.created_at.desc())
        return list(self._db.scalars(statement).all())
