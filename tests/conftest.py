import os
import sys
import pathlib

# Add project root to sys.path so `import backend` works whether the test
# runner is invoked as `pytest` or `python -m pytest`.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Force MOCK_MODE before any backend module is imported so Config() picks it up.
os.environ["MOCK_MODE"] = "1"
