from __future__ import annotations

import os

os.environ["DATA_DIR"] = "data_demo"  # noqa: E402
import panel as pn  # noqa: E402
from regional_dashboard import RegionalDashboard  # noqa: E402

import logging  # noqa: E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_dashboard_from_demo_data():
    logger.debug("Creating dashboard instance from data_demo...")
    instance = RegionalDashboard()
    instance.create_dashboard()

    assert instance.dashboard is not None

    tab_contents = [tab[-1] for tab in instance.tabs]
    last_tab = tab_contents[-1]
    assert any(
        isinstance(child, pn.pane.Markdown) and "## No extreme to show" in child.object
        for child in last_tab[0]
    )
