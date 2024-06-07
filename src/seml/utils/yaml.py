import yaml

from seml.utils.errors import ConfigError


class YamlUniqueLoader(yaml.FullLoader):
    """
    Custom YAML loader that disallows duplicate keys

    From https://github.com/encukou/naucse_render/commit/658197ed142fec2fe31574f1ff24d1ff6d268797
    Workaround for PyYAML issue: https://github.com/yaml/pyyaml/issues/165
    This disables some uses of YAML merge (`<<`)
    """


def construct_mapping(loader, node, deep=False):
    """Construct a YAML mapping node, avoiding duplicates"""
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"Found duplicate keys: '{key}'")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


YamlUniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_mapping,
)
