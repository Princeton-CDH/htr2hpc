"""Tests for the cleanup_exports management command."""
import datetime
import os

import pytest
from django.core.management import call_command
from django.test import override_settings

from htr2hpc.management.commands.cleanup_exports import get_old_exports


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


@pytest.mark.django_db
def test_deletes_old_export_files(media_root, capsys):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)  # 35 days

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    assert not old_file.exists()


@pytest.mark.django_db
def test_keeps_recent_export_files(media_root):
    recent_file = media_root / "users" / "42" / "export_doc1_test_alto_20240601.zip"
    make_file(recent_file, age_hours=120)  # 5 days

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    assert recent_file.exists()


@pytest.mark.django_db
def test_dry_run_does_not_delete(media_root):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)  # 35 days

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports", dry_run=True)

    assert old_file.exists()


@pytest.mark.django_db
def test_ignores_non_export_files(media_root):
    other_file = media_root / "users" / "42" / "manifest.json"
    make_file(other_file, age_hours=840)  # 35 days

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    assert other_file.exists()


@pytest.mark.django_db
def test_respects_custom_retention(media_root):
    # 240 hours (10 days) old — older than 168-hour (1 week) but newer than 720-hour (30 day)
    borderline_file = media_root / "users" / "42" / "export_doc1_test_alto_20240601.zip"
    make_file(borderline_file, age_hours=240)

    with override_settings(EXPORT_FILE_RETENTION=720, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")
    assert borderline_file.exists(), "should be kept under 720-hour (30-day) retention"

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")
    assert not borderline_file.exists(), "should be deleted under 168-hour (1-week) retention"


@pytest.mark.django_db
def test_missing_users_dir_returns_zeros(tmp_path, capsys):
    # MEDIA_ROOT exists but has no users/ subdirectory
    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(tmp_path)):
        call_command("cleanup_exports")

    captured = capsys.readouterr()
    assert "nothing to clean up" in captured.out


@pytest.mark.django_db
def test_reports_bytes_freed(media_root, capsys):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    old_file.write_bytes(b"x" * 1024)
    mtime = (datetime.datetime.now() - datetime.timedelta(hours=840)).timestamp()
    os.utime(old_file, (mtime, mtime))

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    captured = capsys.readouterr()
    assert "1024 bytes" in captured.out


@pytest.mark.django_db
def test_deletes_files_across_multiple_users(media_root):
    user2_dir = media_root / "users" / "99"
    user2_dir.mkdir()

    old1 = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    old2 = user2_dir / "export_doc2_test_pagexml_20240101.zip"
    make_file(old1, age_hours=840)  # 35 days
    make_file(old2, age_hours=840)  # 35 days

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    assert not old1.exists()
    assert not old2.exists()


@pytest.mark.django_db
def test_unlink_error_is_skipped(media_root, capsys, mocker):
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    make_file(old_file, age_hours=840)

    mocker.patch("pathlib.Path.unlink", side_effect=OSError("permission denied"))

    with override_settings(EXPORT_FILE_RETENTION=168, MEDIA_ROOT=str(media_root)):
        call_command("cleanup_exports")

    assert old_file.exists()
    captured = capsys.readouterr()
    assert "Deleted 0 export file(s)" in captured.out


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


def test_get_old_exports(media_root):
    """Unit test for get_old_exports() helper directly."""
    old_file = media_root / "users" / "42" / "export_doc1_test_alto_20240101.zip"
    recent_file = media_root / "users" / "42" / "export_doc2_test_alto_20240601.zip"
    make_file(old_file, age_hours=840)
    make_file(recent_file, age_hours=10)

    cutoff = datetime.datetime.now() - datetime.timedelta(hours=168)
    results = list(get_old_exports(media_root / "users", cutoff))

    assert len(results) == 1
    assert results[0][0] == old_file

