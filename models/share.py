from pathlib import Path


ROOT = Path(__file__).parent.parent
ESM_PATH = ROOT / "dataset" / "ESM.npy"
PROT_PATH = ROOT / "dataset" / "Prot.npy"
LABELS_PATH = ROOT / "dataset" / "Labels.npy"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
