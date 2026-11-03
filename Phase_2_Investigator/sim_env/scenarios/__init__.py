"""Crime-scene scenario definitions (YAML) and a seeded random scenario generator."""
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"          # simulator scenarios (houses / simple room)
ANALYTIC_DIR = Path(__file__).parent / "analytic"       # stand-alone analytic rooms (no simulator)
GENERATED_DIR = Path(__file__).parent / "generated"     # random_sim.py output (gitignored)


def scenario_paths():
    """Simulator scenarios, in order (run_sim.sh --scenario N)."""
    return sorted(CONFIG_DIR.glob("*.yaml"))


def analytic_paths():
    """Stand-alone analytic-room scenarios used by the pure-Python tests, training and evaluation."""
    return sorted(ANALYTIC_DIR.glob("*.yaml"))


def resolve_scenario(spec):
    """Path of a scenario given its 1-based number, its id or a YAML path."""
    spec = str(spec)
    paths = scenario_paths()
    if spec.isdigit():
        n = int(spec)
        if not 1 <= n <= len(paths):
            raise ValueError(f"scenario {n}: choose 1-{len(paths)}")
        return paths[n - 1]
    for p in paths + sorted(GENERATED_DIR.glob("*.yaml")):
        if p.stem == spec or p.stem.split("_")[0] == spec:
            return p
    p = Path(spec)
    if p.exists():
        return p
    raise ValueError(f"unknown scenario {spec!r}; known: {[p.stem for p in paths]}")
