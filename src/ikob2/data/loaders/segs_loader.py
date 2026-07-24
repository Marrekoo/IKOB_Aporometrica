from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ikob2.data.readers.csv_reader import read_csv_float32
from ikob2.core.numerics import DTYPE


INCOME_CLASSES = ["laag", "middellaag", "middelhoog", "hoog"]


@dataclass
class SegsData:
    ikob_id: str
    scenario: str
    population_by_income: dict[str, np.ndarray]
    jobs_by_income: dict[str, np.ndarray]
    n_zones: int


def load_segs(base_path: Path, ikob_id: str, scenario: str) -> SegsData:
    """
    Load SEGS data for one IKOB area and one scenario
    (e.g. 2018, 2030H, 2040H, 2040L).
    """

    scenario_path = base_path / ikob_id / "SEGS" / scenario

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario folder not found: {scenario_path}")

    pop_file = scenario_path / "Beroepsbevolking_inkomensklasse.csv"
    jobs_file = scenario_path / "Arbeidsplaatsen_inkomensklasse.csv"

    pop_matrix = read_csv_float32(pop_file, has_index_column=True)
    jobs_matrix = read_csv_float32(jobs_file, has_index_column=True)

    # Validate shape consistency
    if pop_matrix.shape != jobs_matrix.shape:
        raise ValueError(
            f"Population and jobs shape mismatch in {ikob_id} {scenario}"
        )

    n_zones, n_income = pop_matrix.shape

    if n_income != len(INCOME_CLASSES):
        raise ValueError(
            f"Expected {len(INCOME_CLASSES)} income columns, "
            f"found {n_income} in {pop_file}"
        )

    population = {
        income: pop_matrix[:, i].astype(DTYPE, copy=False)
        for i, income in enumerate(INCOME_CLASSES)
    }

    jobs = {
        income: jobs_matrix[:, i].astype(DTYPE, copy=False)
        for i, income in enumerate(INCOME_CLASSES)
    }

    return SegsData(
        ikob_id=ikob_id,
        scenario=scenario,
        population_by_income=population,
        jobs_by_income=jobs,
        n_zones=n_zones,
    )