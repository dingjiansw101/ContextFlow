"""Validation for the shared LIBERO seen/unseen task split."""

from openpi_client.libero_task_split import LIBERO_UNSEEN_TASKS


def test_unseen_tasks_are_unique():
    assert len(LIBERO_UNSEEN_TASKS) == 8
    assert len(LIBERO_UNSEEN_TASKS) == len(set(LIBERO_UNSEEN_TASKS))
