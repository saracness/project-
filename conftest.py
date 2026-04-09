"""
pytest configuration.

- Sets MPLBACKEND=Agg for all test runs so matplotlib never tries to open
  a display window.
- Marks test files that require a real display as 'gui'; they are skipped
  automatically in headless CI and can be run locally with --gui.
"""
import os
import pytest

# Force non-interactive matplotlib backend before any test imports it.
os.environ.setdefault("MPLBACKEND", "Agg")

# Test files known to require a live display or pygame window.
_GUI_TESTS = {
    "test_button_functionality",
    "test_click_ai",
    "test_spawn_simple",
    "test_launcher",
    "test_interactive",
}


def pytest_addoption(parser):
    parser.addoption(
        "--gui",
        action="store_true",
        default=False,
        help="Include tests that require a display / pygame window.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gui"):
        return  # run everything

    skip = pytest.mark.skip(reason="requires display -- pass --gui to include")
    for item in items:
        for name in _GUI_TESTS:
            if name in item.nodeid:
                item.add_marker(skip)
                break
