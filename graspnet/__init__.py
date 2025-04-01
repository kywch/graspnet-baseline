from pathlib import Path

from .network import GraspNet, pred_decode

BASE_DIR = Path("/".join(__path__[0].split("/")[:-1]))
