"""Tests for the usage report export functions in exports.py."""

import itertools
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest
from core.models import Document, DocumentPart, OcrModel, Project
from django.utils import timezone
from reporting.models import TaskReport

from htr2hpc.exports import (
    HPC_SEGTRAIN_METHOD,
    HPC_TRAIN_METHOD,
    REPORT_MONTH,
    REPORT_TOTAL,
    REPORT_WEEK,
    USAGE_REPORT_FIELDS,
    VM_SEGMENT_METHOD,
    VM_TRANSCRIBE_METHOD,
    get_usage_report,
    report_date_range,
    report_row,
)

# a fixed "today" for the report: Monday 2026-11-09
TODAY = date(2026, 11, 9)


def at(year, month, day, hour=12):
    """Timezone-aware datetime in the site's configured timezone."""
    return datetime(year, month, day, hour, tzinfo=timezone.get_current_timezone())


_task_ids = itertools.count(1)


def add_report(
    user, started_at, method=VM_SEGMENT_METHOD, task_id=None, minutes=10, cpu=1.0
):
    # default to a unique task_id, i.e. a job with a single report
    if task_id is None:
        task_id = f"task-{next(_task_ids)}"
    return TaskReport.objects.create(
        user=user,
        method=method,
        task_id=task_id,
        started_at=started_at,
        done_at=started_at + timedelta(minutes=minutes),
        cpu_cost=cpu,
    )


@pytest.fixture
def today():
    with patch("htr2hpc.exports.timezone.localdate", return_value=TODAY):
        yield TODAY


@pytest.fixture
def alice(django_user_model):
    return django_user_model.objects.create_user(
        username="alice", email="alice@example.com"
    )


@pytest.fixture
def bob(django_user_model):
    return django_user_model.objects.create_user(
        username="bob", email="bob@example.com"
    )


def rows_for(start, report_type):
    return [
        report_row(row, report_type) for row in get_usage_report(start, report_type)
    ]


# ---------------------------------------------------------------------------
# report_date_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "report_type", "expected"),
    [
        # totals: everything from the start date, no end
        (date(2026, 8, 25), REPORT_TOTAL, (date(2026, 8, 25), None)),
        # month: partial first month skipped; current month excluded
        (date(2026, 8, 25), REPORT_MONTH, (date(2026, 9, 1), date(2026, 11, 1))),
        (date(2026, 9, 1), REPORT_MONTH, (date(2026, 9, 1), date(2026, 11, 1))),
        (date(2026, 12, 15), REPORT_MONTH, (date(2027, 1, 1), date(2026, 11, 1))),
        # week: starts on the next Monday; current week excluded
        (date(2026, 10, 1), REPORT_WEEK, (date(2026, 10, 5), date(2026, 11, 9))),
        (date(2026, 10, 5), REPORT_WEEK, (date(2026, 10, 5), date(2026, 11, 9))),
    ],
)
def test_report_date_range(start, report_type, expected):
    assert report_date_range(start, report_type, TODAY) == expected


@pytest.mark.parametrize(
    ("today", "expected_end"),
    [
        # run on a Monday: the week that just ended is included
        (date(2026, 11, 9), date(2026, 11, 9)),
        # run mid-week (Wednesday): the current week is still excluded
        (date(2026, 11, 11), date(2026, 11, 9)),
    ],
)
def test_report_date_range_week_end(today, expected_end):
    _, end = report_date_range(date(2026, 10, 5), REPORT_WEEK, today)
    assert end == expected_end


# ---------------------------------------------------------------------------
# get_usage_report
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_total_report(today, alice):
    add_report(alice, at(2026, 9, 3), VM_SEGMENT_METHOD, minutes=10, cpu=2.0)
    add_report(alice, at(2026, 10, 20), VM_TRANSCRIBE_METHOD, minutes=5, cpu=0.5)
    # before the start date: excluded
    add_report(alice, at(2026, 8, 1))
    OcrModel.objects.create(
        name="model 1", owner=alice, file_size=1000, job=OcrModel.MODEL_JOB_RECOGNIZE
    )
    OcrModel.objects.create(
        name="model 2", owner=alice, file_size=24, job=OcrModel.MODEL_JOB_SEGMENT
    )

    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row == {
        "username": "alice",
        "report_type": "total",
        "report_start": date(2026, 9, 1),
        "report_end": date(2026, 10, 20),
        "disk_usage_bytes": 1024,
        "task_count": 2,
        "runtime_seconds": 15 * 60,
        "cpu_usage": 2.5,
        "vm_segment_tasks": 1,
        "vm_transcribe_tasks": 1,
        "hpc_segtrain_tasks": 0,
        "hpc_train_tasks": 0,
    }


def add_document_images(owner, sizes, name="doc"):
    """Create a document with one part per image size, owned by owner."""
    project = Project.objects.create(name=name, owner=owner)
    document = Document.objects.create(name=name, owner=owner, project=project)
    for i, size in enumerate(sizes):
        DocumentPart.objects.create(
            document=document, order=i, image=f"{name}-{i}.png", image_file_size=size
        )
    return document


@pytest.mark.django_db
def test_total_report_disk_usage_includes_images(today, alice):
    add_report(alice, at(2026, 9, 3))
    OcrModel.objects.create(
        name="model 1", owner=alice, file_size=1000, job=OcrModel.MODEL_JOB_RECOGNIZE
    )
    OcrModel.objects.create(
        name="model 2", owner=alice, file_size=24, job=OcrModel.MODEL_JOB_SEGMENT
    )
    # several documents and parts; neither source should multiply the other
    add_document_images(alice, [100, 200, 300], name="doc1")
    add_document_images(alice, [4000], name="doc2")

    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["disk_usage_bytes"] == 1024 + 4600


@pytest.mark.django_db
def test_total_report_disk_usage_images_only(today, alice, bob):
    add_report(alice, at(2026, 9, 3))
    add_report(bob, at(2026, 9, 3))
    add_document_images(alice, [500, 250])
    # another user's images are not counted
    add_document_images(bob, [9999], name="bobdoc")

    rows = {r["username"]: r for r in rows_for(date(2026, 9, 1), REPORT_TOTAL)}
    assert rows["alice"]["disk_usage_bytes"] == 750
    assert rows["bob"]["disk_usage_bytes"] == 9999


@pytest.mark.django_db
def test_total_report_includes_current_month(today, alice):
    add_report(alice, at(2026, 11, 8))
    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["report_end"] == date(2026, 11, 8)


@pytest.mark.django_db
def test_total_report_no_disk_usage(today, alice):
    add_report(alice, at(2026, 9, 3))
    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["disk_usage_bytes"] == 0


@pytest.mark.django_db
def test_monthly_report_completed_months_only(today, alice):
    # start 2026-08-25: partial August is skipped, November (current) excluded
    add_report(alice, at(2026, 8, 28))
    add_report(alice, at(2026, 9, 2))
    add_report(alice, at(2026, 9, 30, hour=20))
    add_report(alice, at(2026, 10, 15))
    add_report(alice, at(2026, 11, 2))

    rows = rows_for(date(2026, 8, 25), REPORT_MONTH)
    assert [(r["report_start"], r["report_end"], r["task_count"]) for r in rows] == [
        (date(2026, 9, 1), date(2026, 9, 30), 2),
        (date(2026, 10, 1), date(2026, 10, 15), 1),
    ]
    # disk usage is only reported for totals
    assert {r["disk_usage_bytes"] for r in rows} == {""}
    assert {r["report_type"] for r in rows} == {"month"}


@pytest.mark.django_db
def test_monthly_report_across_year_boundary(alice):
    add_report(alice, at(2026, 12, 31, hour=23))
    add_report(alice, at(2027, 1, 1, hour=0))

    with patch("htr2hpc.exports.timezone.localdate", return_value=date(2027, 2, 10)):
        rows = rows_for(date(2026, 12, 1), REPORT_MONTH)
    assert [(r["report_start"], r["report_end"]) for r in rows] == [
        (date(2026, 12, 1), date(2026, 12, 31)),
        (date(2027, 1, 1), date(2027, 1, 1)),
    ]


@pytest.mark.django_db
def test_weekly_report_completed_weeks_only(today, alice):
    # Monday-start weeks; current week (from Mon 2026-11-09) is excluded
    add_report(alice, at(2026, 10, 4))  # Sunday before the first full week
    add_report(alice, at(2026, 10, 5))  # Monday
    add_report(alice, at(2026, 10, 7))  # Wednesday, same week
    add_report(alice, at(2026, 11, 8))  # Sunday, last completed week
    add_report(alice, at(2026, 11, 9))  # Monday of the current week

    rows = rows_for(date(2026, 10, 1), REPORT_WEEK)
    assert [(r["report_start"], r["report_end"], r["task_count"]) for r in rows] == [
        (date(2026, 10, 5), date(2026, 10, 7), 2),
        (date(2026, 11, 2), date(2026, 11, 8), 1),
    ]
    assert {r["report_type"] for r in rows} == {"week"}


@pytest.mark.django_db
def test_hpc_jobs_counted_by_task_id(today, alice):
    # one training job on 3 pages creates 3 task reports with the same task_id
    for _ in range(3):
        add_report(alice, at(2026, 9, 10), HPC_TRAIN_METHOD, task_id="train-1")
    add_report(alice, at(2026, 9, 11), HPC_TRAIN_METHOD, task_id="train-2")
    add_report(alice, at(2026, 9, 12), HPC_SEGTRAIN_METHOD, task_id="seg-1")
    add_report(alice, at(2026, 9, 12), HPC_SEGTRAIN_METHOD, task_id="seg-1")

    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["hpc_train_tasks"] == 2
    assert row["hpc_segtrain_tasks"] == 1


@pytest.mark.django_db
def test_multi_page_job_counted_once(today, alice):
    # one training job on 10 pages: 10 task reports sharing a task_id,
    # each with the same runtime and cpu cost
    for _ in range(10):
        add_report(
            alice, at(2026, 9, 10), HPC_TRAIN_METHOD, task_id="t1", minutes=30, cpu=4.0
        )
    # a separate job
    add_report(alice, at(2026, 9, 11), VM_SEGMENT_METHOD, task_id="s1", minutes=10)

    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["task_count"] == 2
    assert row["runtime_seconds"] == 40 * 60
    assert row["cpu_usage"] == 5.0
    assert row["hpc_train_tasks"] == 1


@pytest.mark.django_db
def test_reports_without_task_id_not_counted(today, alice):
    # e.g. ontology import: a report created outside of celery
    for task_id in [None, ""]:
        TaskReport.objects.create(
            user=alice, task_id=task_id, started_at=at(2026, 9, 10)
        )
    add_report(alice, at(2026, 9, 11))
    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["task_count"] == 1


@pytest.mark.django_db
def test_job_assigned_to_period_of_first_report(today, alice):
    # job starts just before midnight at the end of September; one page report
    # is started after midnight, in October
    add_report(alice, at(2026, 9, 30, 23), HPC_TRAIN_METHOD, task_id="t1")
    add_report(alice, at(2026, 10, 1, 0), HPC_TRAIN_METHOD, task_id="t1")
    rows = rows_for(date(2026, 9, 1), REPORT_MONTH)
    assert [(r["report_start"], r["task_count"]) for r in rows] == [
        (date(2026, 9, 1), 1)
    ]


@pytest.mark.django_db
def test_job_started_before_start_date_excluded(today, alice):
    add_report(alice, at(2026, 8, 31, 23), HPC_TRAIN_METHOD, task_id="t1")
    add_report(alice, at(2026, 9, 1, 0), HPC_TRAIN_METHOD, task_id="t1")
    assert rows_for(date(2026, 9, 1), REPORT_TOTAL) == []


@pytest.mark.django_db
def test_monthly_multi_page_job_counted_once(today, alice):
    for _ in range(3):
        add_report(alice, at(2026, 9, 10), HPC_TRAIN_METHOD, task_id="t1", minutes=5)
    [row] = rows_for(date(2026, 9, 1), REPORT_MONTH)
    assert row["task_count"] == 1
    assert row["runtime_seconds"] == 5 * 60


@pytest.mark.django_db
def test_users_without_activity_have_no_rows(today, alice, bob, django_user_model):
    # carol has an account but no tasks at all
    django_user_model.objects.create_user(username="carol", email="carol@example.com")
    # alice is only active in September, bob only in October
    add_report(alice, at(2026, 9, 3))
    add_report(bob, at(2026, 10, 3))

    totals = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert [r["username"] for r in totals] == ["alice", "bob"]

    # no row for alice in October or bob in September
    monthly = rows_for(date(2026, 9, 1), REPORT_MONTH)
    assert [(r["username"], r["report_start"]) for r in monthly] == [
        ("alice", date(2026, 9, 1)),
        ("bob", date(2026, 10, 1)),
    ]


@pytest.mark.django_db
def test_report_row_null_runtime_and_cpu(today, alice):
    TaskReport.objects.create(
        user=alice, method=VM_SEGMENT_METHOD, task_id="t1", started_at=at(2026, 9, 3)
    )
    [row] = rows_for(date(2026, 9, 1), REPORT_TOTAL)
    assert row["runtime_seconds"] == 0
    assert row["cpu_usage"] == 0


@pytest.mark.django_db
@pytest.mark.parametrize("report_type", [REPORT_TOTAL, REPORT_MONTH, REPORT_WEEK])
def test_report_rows_have_all_fields(today, alice, report_type):
    add_report(alice, at(2026, 10, 6))
    [row] = rows_for(date(2026, 9, 1), report_type)
    assert list(row) == USAGE_REPORT_FIELDS
