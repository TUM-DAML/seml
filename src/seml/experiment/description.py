from __future__ import annotations

from typing import Mapping

from seml.experiment.config import (
    escape_non_interpolated_dollars,
    requires_interpolation,
)


def resolve_description(description: str, config: Mapping) -> str:
    import uuid

    from omegaconf import OmegaConf

    if not requires_interpolation({'seml.description': description}):
        return description

    # omegaconf can only resolve dicts that refers to its own values
    # so we add the description string to the config
    config = escape_non_interpolated_dollars(config, [])
    key = str(uuid.uuid4())
    omg_config = OmegaConf.create(
        {key: description, **config}, flags={'allow_objects': True}
    )
    return OmegaConf.to_container(omg_config, resolve=True)[key]  # type: ignore
