from __future__ import annotations

from seml.document import ExperimentDoc


def resolve_description(description: str, config: dict | ExperimentDoc) -> str:
    import uuid

    from omegaconf import OmegaConf

    # omegaconf can only resolve dicts that refers to its own values
    # so we add the description string to the config
    key = str(uuid.uuid4())
    omg_config = OmegaConf.create(
        {key: description, **config}, flags={'allow_objects': True}
    )
    return OmegaConf.to_container(omg_config, resolve=True)[key]  # type: ignore
