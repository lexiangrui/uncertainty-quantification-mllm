import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(METHOD_ROOT / "src"))
