from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"
FIG, TAB = REPORTS / "figures", REPORTS / "tables"
for p in (DATA / "raw", DATA / "processed", MODELS, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)


def cfg():
    return yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))
