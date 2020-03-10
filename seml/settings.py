from munch import munchify
from pathlib import Path

__all__ = ("SETTINGS",)


SETTINGS = munchify(
    {
        "DATABASE": {
            # location of the MongoDB config. Default: $HOME/.config/seml/monogdb.config
            "MONGODB_CONFIG_PATH": f'{str(Path.home())}/.config/seml/mongodb.config'
        },
    }
)
