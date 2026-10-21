"""Crime-scene scenario definitions (YAML) and a seeded random scenario generator."""
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def scenario_paths():
    return sorted(CONFIG_DIR.glob("*.yaml"))
