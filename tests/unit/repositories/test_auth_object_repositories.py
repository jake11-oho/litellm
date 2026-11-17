from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.repositories import rust_database
from litellm.repositories.access_group_repository import AccessGroupRepository
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.repositories.project_repository import ProjectRepository
from litellm.repositories.rust_database import NoDatabaseConnected
from litellm.repositories.tag_repository import TagRepository
from tests.unit.proxy.rust_auth_stores import RustAuthStores, install_rust_auth_stores

PROJECT: Final = LiteLLM_ProjectTable(project_id="project-1", project_alias="alpha")
TAG: Final = LiteLLM_TagTable(tag_name="tag-a", models=["gpt-4o"])
READER_GROUP: Final = LiteLLM_AccessGroupTable(access_group_id="group-1", access_group_name="reader")
WRITER_GROUP: Final = LiteLLM_AccessGroupTable(access_group_id="group-1", access_group_name="writer")
READER_PERMISSION: Final = LiteLLM_ObjectPermissionTable(object_permission_id="perm-1", models=["reader"])
WRITER_PERMISSION: Final = LiteLLM_ObjectPermissionTable(object_permission_id="perm-1", models=["writer"])


@pytest.fixture
def stores(monkeypatch: pytest.MonkeyPatch) -> RustAuthStores:
    installed: Final = install_rust_auth_stores(monkeypatch)
    installed.project.seed(PROJECT.project_id, PROJECT)
    installed.tag.seed(TAG.tag_name, TAG)
    installed.access_group.reader[READER_GROUP.access_group_id] = READER_GROUP
    installed.access_group.writer[WRITER_GROUP.access_group_id] = WRITER_GROUP
    installed.object_permission.reader[READER_PERMISSION.object_permission_id] = READER_PERMISSION
    installed.object_permission.writer[WRITER_PERMISSION.object_permission_id] = WRITER_PERMISSION
    return installed


@pytest.mark.parametrize(
    ("project_id", "expected"),
    [
        pytest.param("project-1", PROJECT, id="found"),
        pytest.param("project-missing", None, id="missing"),
    ],
)
async def test_project_find_for_auth(
    stores: RustAuthStores, project_id: str, expected: LiteLLM_ProjectTable | None
) -> None:
    assert await ProjectRepository(None).find_for_auth(project_id) == expected
    assert stores.project.lookups == [(project_id, False)]


async def test_project_find_for_auth_without_a_database_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_database, "repository", lambda kind: None)

    assert await ProjectRepository(None).find_for_auth("project-1") is None


async def test_tag_names(stores: RustAuthStores) -> None:
    assert await TagRepository(None).names(5) == ("tag-a",)
    assert stores.tag.name_listings == [(5, False)]


async def test_tag_find_many_for_auth_keeps_the_tags_that_exist(stores: RustAuthStores) -> None:
    assert await TagRepository(None).find_many_for_auth(("tag-a", "tag-missing")) == (TAG,)
    assert stores.tag.batches == [(("tag-a", "tag-missing"), False)]


@pytest.mark.parametrize(
    ("use_writer", "expected"),
    [
        pytest.param(False, READER_GROUP, id="reader"),
        pytest.param(True, WRITER_GROUP, id="writer"),
    ],
)
async def test_access_group_find_for_auth_reads_the_side_asked_for(
    stores: RustAuthStores, use_writer: bool, expected: LiteLLM_AccessGroupTable
) -> None:
    assert await AccessGroupRepository(None).find_for_auth("group-1", use_writer=use_writer) == expected
    assert stores.access_group.lookups == [("group-1", use_writer)]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(READER_GROUP, READER_GROUP, id="found"),
        pytest.param(None, None, id="missing"),
    ],
)
async def test_access_group_find_by_name(
    row: LiteLLM_AccessGroupTable | None, expected: LiteLLM_AccessGroupTable | None
) -> None:
    find_unique: Final = AsyncMock(return_value=row)
    prisma: Final = SimpleNamespace(
        db=SimpleNamespace(litellm_accessgrouptable=SimpleNamespace(find_unique=find_unique))
    )

    assert await AccessGroupRepository(prisma).find_by_name("reader") == expected
    find_unique.assert_awaited_once_with(where={"access_group_name": "reader"})


@pytest.mark.parametrize(
    ("use_writer", "expected"),
    [
        pytest.param(False, READER_PERMISSION, id="reader"),
        pytest.param(True, WRITER_PERMISSION, id="writer"),
    ],
)
async def test_object_permission_find_for_auth_reads_the_side_asked_for(
    stores: RustAuthStores, use_writer: bool, expected: LiteLLM_ObjectPermissionTable
) -> None:
    assert await ObjectPermissionRepository(None).find_for_auth("perm-1", use_writer=use_writer) == expected
    assert stores.object_permission.lookups == [("perm-1", use_writer)]


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda: TagRepository(None).names(5), id="tag-names"),
        pytest.param(lambda: TagRepository(None).find_many_for_auth(("tag-a",)), id="tags"),
        pytest.param(lambda: AccessGroupRepository(None).find_for_auth("group-1"), id="access-group"),
        pytest.param(lambda: ObjectPermissionRepository(None).find_for_auth("perm-1"), id="object-permission"),
    ],
)
async def test_required_reads_without_a_database_raise(
    monkeypatch: pytest.MonkeyPatch, read: Callable[[], Awaitable[object]]
) -> None:
    monkeypatch.setattr(rust_database, "repository", lambda kind: None)

    with pytest.raises(NoDatabaseConnected):
        await read()
