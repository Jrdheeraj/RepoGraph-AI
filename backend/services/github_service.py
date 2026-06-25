import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid4

from git import GitCommandError, Repo
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.models.repository import Repository
from backend.repositories.repository_repository import RepositoryRepository


logger = logging.getLogger(__name__)


class InvalidGitHubUrlError(ValueError):
    pass


class RepositoryAlreadyExistsError(ValueError):
    pass


class RepositoryCloneError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubRepositoryInfo:
    github_url: str
    owner: str
    name: str


class GitHubService:
    def __init__(
        self,
        repository_repository: RepositoryRepository,
        temp_repos_dir: Path | None = None,
    ) -> None:
        self._repository_repository = repository_repository
        self._temp_repos_dir = temp_repos_dir or Path(__file__).resolve().parents[1] / "temp_repos"

    def import_repository(self, github_url: str) -> Repository:
        repo_info = self._parse_github_url(github_url)

        existing_repository = self._repository_repository.get_by_github_url(repo_info.github_url)
        if existing_repository is not None:
            logger.warning(
                "repository_import_duplicate",
                extra={
                    "repository_id": str(existing_repository.id),
                    "github_url": repo_info.github_url,
                },
            )
            raise RepositoryAlreadyExistsError("Repository has already been imported.")

        repository_id = uuid4()
        clone_path = self._temp_repos_dir / str(repository_id)

        logger.info(
            "repository_import_started",
            extra={
                "repository_id": str(repository_id),
                "github_url": repo_info.github_url,
                "clone_path": str(clone_path),
            },
        )

        try:
            self._clone_repository(repo_info.github_url, clone_path)
            repository = self._repository_repository.create(
                repository_id=repository_id,
                github_url=repo_info.github_url,
                name=repo_info.name,
                owner=repo_info.owner,
                language=None,
                status="pending",
            )
        except IntegrityError as exc:
            self._remove_clone_path(clone_path)
            logger.exception(
                "repository_import_integrity_error",
                extra={
                    "repository_id": str(repository_id),
                    "github_url": repo_info.github_url,
                },
            )
            raise RepositoryAlreadyExistsError("Repository has already been imported.") from exc
        except SQLAlchemyError:
            self._remove_clone_path(clone_path)
            logger.exception(
                "repository_import_database_error",
                extra={
                    "repository_id": str(repository_id),
                    "github_url": repo_info.github_url,
                },
            )
            raise
        except Exception:
            self._remove_clone_path(clone_path)
            logger.exception(
                "repository_import_failed",
                extra={
                    "repository_id": str(repository_id),
                    "github_url": repo_info.github_url,
                    "clone_path": str(clone_path),
                },
            )
            raise

        logger.info(
            "repository_import_completed",
            extra={
                "repository_id": str(repository.id),
                "github_url": repository.github_url,
            },
        )
        return repository

    def _clone_repository(self, github_url: str, clone_path: Path) -> None:
        self._temp_repos_dir.mkdir(parents=True, exist_ok=True)

        try:
            Repo.clone_from(github_url, clone_path)
        except GitCommandError as exc:
            raise RepositoryCloneError("Failed to clone GitHub repository.") from exc

    def _parse_github_url(self, github_url: str) -> GitHubRepositoryInfo:
        normalized_url = github_url.strip()
        parsed_url = urlparse(normalized_url)

        if parsed_url.scheme not in {"http", "https"}:
            raise InvalidGitHubUrlError("GitHub URL must use http or https.")

        if parsed_url.netloc.lower() != "github.com":
            raise InvalidGitHubUrlError("GitHub URL must point to github.com.")

        path_parts = [part for part in parsed_url.path.strip("/").split("/") if part]
        if len(path_parts) < 2:
            raise InvalidGitHubUrlError("GitHub URL must include an owner and repository name.")

        owner = path_parts[0]
        name = path_parts[1].removesuffix(".git")

        if not owner or not name:
            raise InvalidGitHubUrlError("GitHub URL must include an owner and repository name.")

        canonical_url = f"https://github.com/{owner}/{name}"
        return GitHubRepositoryInfo(github_url=canonical_url, owner=owner, name=name)

    def _remove_clone_path(self, clone_path: Path) -> None:
        if clone_path.exists():
            shutil.rmtree(clone_path)
