"""Tests for the cleanup_exports management command."""
import datetime
import os

import pytest

from htr2hpc.management.commands.cleanup_exports import (
    EXPORT_FILE_RETENTION_DEFAULT,
    delete_old_exports,
)


def make_file(path, age_days):
    """Create a file and set its mtime to age_days ago."""
    path.touch()
    mtime = (datetime.datetime.now() - datetime.timedelta(days=age_days)).timestamp()
    os.utime(path, (mtime, mtime))


@pytest.fixture
def media_root(tmp_path):
    """Set up a fake MEDIA_ROOT with a users directory."""
    user_dir = tmp_path / "users" / "42"
    user_dir.mkdir(parents=True)
    return tmp_path


def test_deletes_old_export_files(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_days=35)

    count, total_bytes = delete_old_exports(str(media_root), EXPORT_FILE_RETENTION_DEFAULT)

    assert not old_file.exists()
    assert count == 1
    assert total_bytes == 0  # empty file


def test_keeps_recent_export_files(media_root):
    recent_file = media_root / "users" / "42" / "export_doc1_test_alto_20240601.zip"
    make_file(recent_file, age_days=5)

    count, total_bytes = delete_old_exports(str(media_root), EXPORT_FILE_RETENTION_DEFAULT)

    assert recent_file.exists()
    assert count == 0
    assert total_bytes == 0


def test_dry_run_does_not_delete(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_days=35)

    count, total_bytes = delete_old_exports(
        str(media_root), EXPORT_FILE_RETENTION_DEFAULT, dry_run=True
    )

    assert old_file.exists()
    assert count == 1


def test_ignores_non_export_files(media_root):
    other_file = media_root / "users" / "42" / "manifest.json"
    make_file(other_file, age_days=35)

    count, _ = delete_old_exports(str(media_root), EXPORT_FILE_RETENTION_DEFAULT)

    assert other_file.exists()
    assert count == 0


def test_respects_custom_retention(media_root):
    # 10 days old — older than 7-day retention but newer than 30-day default
    borderline_file = media_root / "users" / "42" / "export_doc1_test_alto_20240601.zip"
    make_file(borderline_file, age_days=10)

    count_default, _ = delete_old_exports(str(media_root), retention_days=30)
    assert borderline_file.exists(), "should be kept under default 30-day retention"
    assert count_default == 0

    count_short, _ = delete_old_exports(str(media_root), retention_days=7)
    assert not borderline_file.exists(), "should be deleted under 7-day retention"
    assert count_short == 1


def test_missing_users_dir_returns_zeros(tmp_path):
    # MEDIA_ROOT exists but has no users/ subdirectory
    count, total_bytes = delete_old_exports(str(tmp_path), EXPORT_FILE_RETENTION_DEFAULT)
    assert count == 0
    assert total_bytes == 0


def test_reports_bytes_freed(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    old_file.write_bytes(b"x" * 1024)
    mtime = (datetime.datetime.now() - datetime.timedelta(days=35)).timestamp()
    os.utime(old_file, (mtime, mtime))

    _, total_bytes = delete_old_exports(str(media_root), EXPORT_FILE_RETENTION_DEFAULT)

    assert total_bytes == 1024


def test_deletes_files_across_multiple_users(media_root):
    user2_dir = media_root / "users" / "99"
    user2_dir.mkdir()

    old1 = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    old2 = user2_dir / "export_doc2_test_pagexml_20240101.zip"
    make_file(old1, age_days=35)
    make_file(old2, age_days=35)

    count, _ = delete_old_exports(str(media_root), EXPORT_FILE_RETENTION_DEFAULT)

    assert not old1.exists()
    assert not old2.exists()
    assert count == 2
