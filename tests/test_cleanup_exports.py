"""Tests for the cleanup_exports management command."""
import datetime
import os

import pytest
from django.test import override_settings
from django.core.management import call_command

from htr2hpc.management.commands.cleanup_exports import (
    delete_old_exports,
)


def make_file(path, age_hours):
    """Create a file and set its mtime to age_hours ago."""
    path.touch()
    mtime = (datetime.datetime.now() - datetime.timedelta(hours=age_hours)).timestamp()
    os.utime(path, (mtime, mtime))


@pytest.fixture
def media_root(tmp_path):
    """Set up a fake MEDIA_ROOT with a users directory."""
    user_dir = tmp_path / "users" / "42"
    user_dir.mkdir(parents=True)
    return tmp_path


def test_deletes_old_export_files(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)  # 35 days

    count, total_bytes = delete_old_exports(str(media_root), 168)

    assert not old_file.exists()
    assert count == 1
    assert total_bytes == 0  # empty file


def test_keeps_recent_export_files(media_root):
    recent_file = media_root / "users" / "42" / "export_doc1_test_alto_20240601.zip"
    make_file(recent_file, age_hours=120)  # 5 days

    count, total_bytes = delete_old_exports(str(media_root), 168)

    assert recent_file.exists()
    assert count == 0
    assert total_bytes == 0


def test_dry_run_does_not_delete(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)  # 35 days

    count, _ = delete_old_exports(str(media_root), 168, dry_run=True)

    assert old_file.exists()
    assert count == 1


def test_ignores_non_export_files(media_root):
    other_file = media_root / "users" / "42" / "manifest.json"
    make_file(other_file, age_hours=840)  # 35 days

    count, _ = delete_old_exports(str(media_root), 168)

    assert other_file.exists()
    assert count == 0


def test_respects_custom_retention(media_root):
    # 240 hours (10 days) old — older than 168-hour (1 week) retention but newer than 720-hour (30 day) default
    borderline_file = media_root / "users" / "42" / "export_doc1_test_alto_20240601.zip"
    make_file(borderline_file, age_hours=240)

    count_default, _ = delete_old_exports(str(media_root), retention_hours=720)
    assert borderline_file.exists(), "should be kept under 720-hour (30-day) retention"
    assert count_default == 0

    count_short, _ = delete_old_exports(str(media_root), retention_hours=168)
    assert not borderline_file.exists(), "should be deleted under 168-hour (1-week) retention"
    assert count_short == 1


def test_missing_users_dir_returns_zeros(tmp_path):
    # MEDIA_ROOT exists but has no users/ subdirectory
    count, total_bytes = delete_old_exports(str(tmp_path), 168)
    assert count == 0
    assert total_bytes == 0


def test_reports_bytes_freed(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    old_file.write_bytes(b"x" * 1024)
    mtime = (datetime.datetime.now() - datetime.timedelta(hours=840)).timestamp()  # 35 days
    os.utime(old_file, (mtime, mtime))

    _, total_bytes = delete_old_exports(str(media_root), 168)

    assert total_bytes == 1024


def test_deletes_files_across_multiple_users(media_root):
    user2_dir = media_root / "users" / "99"
    user2_dir.mkdir()

    old1 = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    old2 = user2_dir / "export_doc2_test_pagexml_20240101.zip"
    make_file(old1, age_hours=840)  # 35 days
    make_file(old2, age_hours=840)  # 35 days

    count, _ = delete_old_exports(str(media_root), 168)

    assert not old1.exists()
    assert not old2.exists()
    assert count == 2


def test_unlink_error_is_skipped(media_root, mocker):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)

    mocker.patch("pathlib.Path.unlink", side_effect=OSError("permission denied"))

    count, total_bytes = delete_old_exports(str(media_root), 168)

    assert old_file.exists()
    assert count == 0
    assert total_bytes == 0


@pytest.mark.django_db
def test_handle_retention_zero(tmp_path, capsys):
    with override_settings(EXPORT_FILE_RETENTION=0, MEDIA_ROOT=str(tmp_path)):
        call_command("cleanup_exports")

    captured = capsys.readouterr()
    assert "set to 0" in captured.out


@pytest.mark.django_db
def test_handle_missing_users_dir(tmp_path, capsys):
    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(tmp_path)):
        call_command("cleanup_exports")

    captured = capsys.readouterr()
    assert "nothing to clean up" in captured.out


@pytest.mark.django_db
def test_handle_deletes_old_exports(media_root, capsys):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    assert not old_file.exists()
    captured = capsys.readouterr()
    assert "Deleted 1 export file(s)" in captured.out


@pytest.mark.django_db
def test_handle_dry_run(media_root, capsys):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports", dry_run=True)

    assert old_file.exists()
    captured = capsys.readouterr()
    assert "Would delete 1 export file(s)" in captured.out

