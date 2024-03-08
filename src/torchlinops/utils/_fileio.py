import logging
from pathlib import Path
import shutil
import sys
from typing import Optional, Literal
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "mkdir",
    "save_yaml_options",
]


def mkdir(root_path, action: Optional[Literal["o", "l", "a"]] = None):
    """
    root_path should be a pathlib.Path
    """
    if root_path.exists():
        # Overwrite?
        if action is not None:
            opt = action
        else:
            opt = input(
                f"{root_path} exists. [O]verwrite/[L]eave/[A]bort? [o/l/a] "
            ).lower()
        if opt == "o":
            logger.warning(f"Overwriting {root_path}.")
            shutil.rmtree(root_path)
        elif opt == "l":
            logger.debug(f"Saving to {root_path}")
        else:
            logger.debug("Aborting.")
            sys.exit()
    Path(root_path).mkdir(parents=True, exist_ok=True)


def save_yaml_options(config_file: Path, opt):
    def noop(self, *args, **kwargs):
        pass

    yaml.emitter.Emitter.process_tag = noop
    with open(config_file, "w") as f:
        yaml.dump(opt, f)
