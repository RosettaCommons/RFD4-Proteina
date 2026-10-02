import logging
import os

import torch
from environs import Env

# Load settings before AtomWorks reads its environment during import.
from rfproteina.utils.env import load_dotenv

load_dotenv()

_env = Env()
_env.read_env()
should_debug = _env.bool("DEBUG", default=False)
# Read by utils/torch.py to gate its NaN assertions.
should_check_nans = _env.bool("NAN_CHECK", default=True)

logger = logging.getLogger("rfproteina")
logger.setLevel(logging.DEBUG if should_debug else logging.INFO)

# cuEquivariance tuning. The fused-kernel switch itself lives in nn/modules/triangle.py.
try:
    if torch.cuda.is_available() and not _env.bool("DISABLE_CUEQUIVARIANCE", default=False):
        import cuequivariance_torch

        os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str("CUEQ_DISABLE_AOT_TUNING", default="1")
        os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str("CUEQ_DEFAULT_CONFIG", default="1")
        logger.info("cuEquivariance is available and will be used.")
except ImportError:
    logger.debug("cuEquivariance unavailable: import failed")

# Registers annotations with atomworks' annotator registry (import side-effect).
import rfproteina.datapipes  # noqa: E402
import rfproteina.utils  # noqa: E402
import rfproteina.utils.standard_annotations  # noqa: E402

__all__ = ["datapipes", "utils"]
