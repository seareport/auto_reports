from __future__ import annotations

import logging

from regional_dashboard import RegionalDashboard

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create and display the dashboard
logger.debug("Creating dashboard instance...")
# Create and display the dashboard
instance = RegionalDashboard()
instance.create_dashboard()
instance.dashboard.servable()
