"""
reconstruction/v3_current/vggt_pipeline/__init__.py
---------------------------------------------------
Local bridge to the separate reconstruction/vggt pipeline module.
"""

import sys
import os

# Add project root so 'reconstruction.vggt' is discoverable without shadowing site-packages
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from reconstruction.vggt import *
