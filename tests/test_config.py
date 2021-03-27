import unittest
import yaml

from seml import config, utils
from seml.errors import ConfigError


class TestParseConfigDicts(unittest.TestCase):

    SIMPLE_CONFIG_WITH_PARAMETER_COLLECTIONS = "resources/config/config_with_parameter_collections.yaml"
    SIMPLE_CONFIG_WITH_PARAMETER_COLLECTIONS_RANDOM = "resources/config/config_with_parameter_collections_random.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_1 = "resources/config/config_with_duplicate_parameters_1.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_2 = "resources/config/config_with_duplicate_parameters_2.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_3 = "resources/config/config_with_duplicate_parameters_3.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_NESTED = "resources/config/config_nested_parameter_collections.yaml"
    CONFIG_WITH_DUPLICATE_RDM_PARAMETERS_2 = "resources/config/config_with_duplicate_random_parameters_1.yaml"
    CONFIG_WITH_ALL_TYPES = "resources/config/config_with_all_types.yaml"

    def load_config_dict(self, path):
        with open(path, 'r') as conf:
            config_dict = config.convert_values(yaml.load(conf, Loader=yaml.FullLoader))
        return config_dict

    def test_convert_parameter_collections(self):
        config_dict = self.load_config_dict(self.SIMPLE_CONFIG_WITH_PARAMETER_COLLECTIONS)
        converted = config.convert_parameter_collections(config_dict)
        expected = {
            "grid": {
                'coll1': {
                    "a": {
                        "type": "choice",
                        "options": [1, 2]
                    }
                }
            },
            "random": {
                "samples": 3,
                "seed": 821,
                "coll1": {
                    "b": {
                        "type": "uniform",
                        "min": 0.0,
                        "max": 0.7,
                        "seed": 333,
                    }
                },
            }
        }
        self.assertEqual(converted, expected)

    def test_unpack_config_dict(self):
        config_dict = self.load_config_dict(self.SIMPLE_CONFIG_WITH_PARAMETER_COLLECTIONS)
        unpacked, next_level = config.unpack_config(config_dict)

        self.assertEqual(next_level, {})

        expected_random = {
            'samples': 3,
            'seed': 821,
            'coll1': {
                'b': {
                    'type': 'uniform',
                    'min': 0.0,
                    'max': 0.7,
                    'seed': 333,
                    # 'samples': 4,
                }
            }
        }

        self.assertEqual(unpacked['random'], expected_random)

    def test_duplicate_parameters(self):
        config_dict = self.load_config_dict(self.CONFIG_WITH_DUPLICATE_PARAMETERS_1)
        with self.assertRaises(ConfigError):
            configs = config.generate_configs(config_dict)

        config_dict = self.load_config_dict(self.CONFIG_WITH_DUPLICATE_PARAMETERS_2)
        with self.assertRaises(ConfigError):
            configs = config.generate_configs(config_dict)

        with self.assertRaises(ConfigError):
            configs = config.read_config(self.CONFIG_WITH_DUPLICATE_PARAMETERS_3)

        config_dict = self.load_config_dict(self.CONFIG_WITH_DUPLICATE_PARAMETERS_NESTED)
        with self.assertRaises(ConfigError):
            configs = config.generate_configs(config_dict)

        config_dict = self.load_config_dict(self.CONFIG_WITH_DUPLICATE_RDM_PARAMETERS_2)
        configs = config.generate_configs(config_dict)
        assert len(configs) == config_dict['random']['samples']

    def test_generate_configs(self):
        config_dict = self.load_config_dict(self.CONFIG_WITH_ALL_TYPES)
        configs = config.generate_configs(config_dict)
        assert len(configs) == 22
        expected_configs = [
            *(5*[{'a': 9999, 'b': 7777, 'c': 1234, 'd': 1.0, 'e': 2.0},
                 {'a': 9999, 'b': 7777, 'c': 5678, 'd': 1.0, 'e': 2.0}]),
            *(3*[{'a': 333, 'b': 444, 'c': 555, 'd': 1.0, 'f': 9199},
                 {'a': 333, 'b': 444, 'c': 555, 'd': 1.0, 'f': 1099},
                 {'a': 333, 'b': 444, 'c': 666, 'd': 1.0, 'f': 9199},
                 {'a': 333, 'b': 444, 'c': 666, 'd': 1.0, 'f': 1099}]),
        ]
        expected_config_hashes = sorted([utils.make_hash(x) for x in expected_configs])
        actual_config_hashes = sorted([utils.make_hash(x) for x in configs])
        assert expected_config_hashes == actual_config_hashes
