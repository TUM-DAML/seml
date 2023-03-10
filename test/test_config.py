import copy
import unittest

import yaml

from seml import config, utils
from seml.add import assemble_slurm_config_dict
from seml.config import read_config
from seml.errors import ConfigError
from seml.settings import SETTINGS
from seml.utils import merge_dicts


class TestParseConfigDicts(unittest.TestCase):

    SIMPLE_CONFIG_WITH_PARAMETER_COLLECTIONS = "resources/config/config_with_parameter_collections.yaml"
    SIMPLE_CONFIG_WITH_PARAMETER_COLLECTIONS_RANDOM = "resources/config/config_with_parameter_collections_random.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_1 = "resources/config/config_with_duplicate_parameters_1.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_2 = "resources/config/config_with_duplicate_parameters_2.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_3 = "resources/config/config_with_duplicate_parameters_3.yaml"
    CONFIG_WITH_DUPLICATE_PARAMETERS_NESTED = "resources/config/config_nested_parameter_collections.yaml"
    CONFIG_WITH_DUPLICATE_RDM_PARAMETERS_2 = "resources/config/config_with_duplicate_random_parameters_1.yaml"
    CONFIG_WITH_ALL_TYPES = "resources/config/config_with_all_types.yaml"
    CONFIG_WITH_EMPTY_DICT = "resources/config/config_with_empty_dictionary.yaml"
    CONFIG_WITH_ZIPPED_PARAMETERS = "resources/config/config_with_zipped_parameters.yaml"
    CONFIG_WITH_GRID = "resources/config/config_with_grid.yaml"
    CONFIG_SLURM_DEFAULT = "resources/config/config_slurm_default.yaml"
    CONFIG_SLURM_DEFAULT_EMPTY_SBATCH = "resources/config/config_slurm_default_empty_sbatch.yaml"
    CONFIG_SLURM_TEMPLATE = "resources/config/config_slurm_template.yaml"
    CONFIG_SLURM_EXPERIMENT = "resources/config/config_slurm_experiment.yaml"

    def load_config_dict(self, path):
        with open(path, 'r') as conf:
            config_dict = config.convert_values(yaml.load(conf, Loader=yaml.FullLoader))
        return config_dict

    def test_config_inheritance(self):
        # Check default config
        seml_config, slurm_config, experiment_config = read_config(self.CONFIG_SLURM_DEFAULT)
        slurm_config = assemble_slurm_config_dict(slurm_config)
        self.assertEqual(slurm_config, SETTINGS.SLURM_DEFAULT)

        # Check default config with empty sbatch options
        seml_config, slurm_config, experiment_config = read_config(self.CONFIG_SLURM_DEFAULT_EMPTY_SBATCH)
        slurm_config = assemble_slurm_config_dict(slurm_config)
        self.assertEqual(slurm_config, SETTINGS.SLURM_DEFAULT)

        # Check default -> template inheritance
        seml_config, slurm_config, experiment_config = read_config(self.CONFIG_SLURM_TEMPLATE)
        slurm_config = assemble_slurm_config_dict(slurm_config)
        target_config = copy.deepcopy(SETTINGS.SLURM_DEFAULT)
        target_config['sbatch_options'] = merge_dicts(target_config['sbatch_options'], SETTINGS.SBATCH_OPTIONS_TEMPLATES.GPU)
        target_config['sbatch_options_template'] = 'GPU'
        self.assertEqual(slurm_config, target_config)

        # Check default -> template -> experiment inheritance
        seml_config, slurm_config, experiment_config = read_config(self.CONFIG_SLURM_EXPERIMENT)
        slurm_config = assemble_slurm_config_dict(slurm_config)
        target_config = copy.deepcopy(SETTINGS.SLURM_DEFAULT)
        target_config['sbatch_options'] = merge_dicts(target_config['sbatch_options'], SETTINGS.SBATCH_OPTIONS_TEMPLATES.GPU)
        target_config['sbatch_options_template'] = 'GPU'
        target_config['sbatch_options']['cpus-per-task'] = 4
        self.assertEqual(slurm_config, target_config)


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

    def test_empty_dictionary(self):
        config_dict = self.load_config_dict(self.CONFIG_WITH_EMPTY_DICT)
        configs = config.generate_configs(config_dict)[0]
        expected_config = {
            'attribute': {
            'test': {}
            }
        }
        self.assertEqual(configs, expected_config)
    
    def test_overwrite_parameters(self):
        config_dict = self.load_config_dict(self.CONFIG_WITH_GRID)
        configs = config.generate_configs(config_dict, {
            'dataset': 'small'
        })
        expected_configs = [
            {
                'dataset': 'small',
                'lr': 0.1
            },
            {
                'dataset': 'small',
                'lr': 0.01
            }
        ]
        self.assertEqual(configs, expected_configs)

    def test_zipped_parameters(self):
        config_dict = self.load_config_dict(self.CONFIG_WITH_ZIPPED_PARAMETERS)
        configs = config.generate_configs(config_dict)
        expected_configs = [
            {
                'attribute': { 'test': 1},
                'learning_rate': 0.0,
                'other_attribute': True
            },
            {
                'attribute': { 'test': 1},
                'learning_rate': 0.0,
                'other_attribute': False
            },
            {
                'attribute': { 'test': 2},
                'learning_rate':1.0,
                'other_attribute': True
            },
            {
                'attribute': { 'test': 2},
                'learning_rate': 1.0,
                'other_attribute': False
            }
        ]
        self.assertEqual(configs, expected_configs)

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
