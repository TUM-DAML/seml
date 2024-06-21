from typing import Dict


def resolve_description(description: str, config: Dict) -> str:
    import uuid

    from omegaconf import OmegaConf

    # omegaconf can only resolve dicts that refers to its own values
    # so we add the description string to the config
    key = str(uuid.uuid4())
    config = OmegaConf.create(
        {key: description, **config}, flags={'allow_objects': True}
    )
    return OmegaConf.to_container(config, resolve=True)[key]
