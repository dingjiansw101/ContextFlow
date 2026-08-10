"""The held-out task list is duplicated across a process boundary; keep the copies honest.

Training excludes ``DEFAULT_LIBERO_TEST_TASK`` via ``remove_task_list``. The LIBERO eval
clients classify the same tasks as unseen using their own ``UNSEEN_TASKS`` constant, because
they run in a separate Python 3.8 environment that cannot import ``openpi``. If the two ever
diverge, a task would be trained on and then scored as unseen — silent contamination that
inflates the generalization numbers with no error anywhere.

The client files import ``libero``, which is not installed in the repo environment, so the
constant is read out of the source with ``ast`` rather than imported.
"""

import ast
import pathlib

import pytest

from openpi.training.config import DEFAULT_LIBERO_TEST_TASK

_CLIENTS = [
    "examples/libero/main_incontext.py",
    "examples/libero/main_incontext_unseen.py",
]


def _unseen_tasks_from_source(path: pathlib.Path) -> set[str]:
    """Extract the UNSEEN_TASKS literal from a client module without importing it."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "UNSEEN_TASKS" for t in node.targets):
            continue
        # UNSEEN_TASKS = frozenset({...})
        value = node.value
        if isinstance(value, ast.Call) and len(value.args) == 1:
            value = value.args[0]
        return set(ast.literal_eval(value))
    raise AssertionError(f"UNSEEN_TASKS not found in {path}")


@pytest.mark.parametrize("client", _CLIENTS)
def test_client_unseen_tasks_match_training_config(client):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    client_tasks = _unseen_tasks_from_source(repo_root / client)
    training_tasks = set(DEFAULT_LIBERO_TEST_TASK)

    assert client_tasks == training_tasks, (
        f"{client} UNSEEN_TASKS disagrees with DEFAULT_LIBERO_TEST_TASK.\n"
        f"  only in client:   {sorted(client_tasks - training_tasks)}\n"
        f"  only in training: {sorted(training_tasks - client_tasks)}"
    )


def test_training_config_has_no_duplicate_held_out_tasks():
    assert len(DEFAULT_LIBERO_TEST_TASK) == len(set(DEFAULT_LIBERO_TEST_TASK))
