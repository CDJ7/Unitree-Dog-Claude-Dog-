"""Physics-based bloodstain spatter generator (pure Python; MuJoCo rendering is added separately)."""
from .generator import generate, load_scenario, save_ground_truth, load_ground_truth, build_surfaces, summary  # noqa: F401
