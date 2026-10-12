"""Project locations shared by plotting and tuning tools."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"


def output_path(filename):

    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    return OUTPUT_DIRECTORY / filename
