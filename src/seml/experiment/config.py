from __future__ import annotations

import ast
import copy
import functools
import itertools  # type: ignore - N.Gao: I don't get this error
import logging
import numbers
import os
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

from seml.document import (
    ExperimentConfig,
    ExperimentDoc,
    SBatchOptions,
    SemlConfig,
    SemlDocBase,
    SemlExperimentFile,
    SemlFileConfig,
    SlurmConfig,
)
from seml.experiment.parameters import (
    cartesian_product_zipped_dict,
    generate_grid,
    sample_random_configs,
    zipped_dict,
)
from seml.experiment.sources import import_exe
from seml.settings import SETTINGS
from seml.utils import (
    Hashabledict,
    drop_typeddict_difference,
    flatten,
    merge_dicts,
    remove_keys_from_nested,
    s_if,
    to_typeddict,
    unflatten,
    working_directory,
)
from seml.utils.errors import ConfigError, ExecutableError

if TYPE_CHECKING:
    import sacred
    from pymongo.collection import Collection

RESERVED_KEYS = ['grid', 'fixed', 'random']


def unpack_config(config):
    config = convert_parameter_collections(config)
    children = {}
    reserved_dict = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        if key not in RESERVED_KEYS:
            children[key] = value
        else:
            if key == 'random':
                if 'samples' not in value:
                    raise ConfigError(
                        'Random parameters must specify "samples", i.e. the number of random samples.'
                    )
                reserved_dict[key] = value
            else:
                reserved_dict[key] = value
    return reserved_dict, children


def extract_parameter_set(input_config: dict, key: str):
    flattened_dict = flatten(input_config.get(key, {}))
    keys = flattened_dict.keys()
    if key != 'fixed':
        keys = [
            '.'.join(k.split('.')[:-1])
            for k in keys
            if flattened_dict[k] != 'parameter_collection'
        ]
    return set(keys)


def convert_parameter_collections(input_config: dict):
    flattened_dict = flatten(input_config)
    parameter_collection_keys = [
        k for k in flattened_dict.keys() if flattened_dict[k] == 'parameter_collection'
    ]
    if len(parameter_collection_keys) > 0:
        logging.warning(
            'Parameter collections are deprecated. Use dot-notation for nested parameters instead.'
        )
    while len(parameter_collection_keys) > 0:
        k = parameter_collection_keys[0]
        del flattened_dict[k]
        # sub1.sub2.type ==> # sub1.sub2
        k = '.'.join(k.split('.')[:-1])
        parameter_collections_params = [
            param_key for param_key in flattened_dict.keys() if param_key.startswith(k)
        ]
        for p in parameter_collections_params:
            if f'{k}.params' in p:
                new_key = p.replace(f'{k}.params', k)
                if new_key in flattened_dict:
                    raise ConfigError(
                        f'Could not convert parameter collections due to key collision: {new_key}.'
                    )
                flattened_dict[new_key] = flattened_dict[p]
                del flattened_dict[p]
        parameter_collection_keys = [
            k
            for k in flattened_dict.keys()
            if flattened_dict[k] == 'parameter_collection'
        ]
    return unflatten(flattened_dict)


def standardize_config(config: dict):
    config = unflatten(flatten(config), levels=[0])
    out_dict = {}
    for k in RESERVED_KEYS:
        if k == 'fixed':
            out_dict[k] = config.get(k, {})
        else:
            out_dict[k] = unflatten(config.get(k, {}), levels=[-1])
    return out_dict


def invert_config(config: dict):
    reserved_sets = [(k, set(config.get(k, {}).keys())) for k in RESERVED_KEYS]
    inverted_config = {}
    for k, params in reserved_sets:
        for p in params:
            L = inverted_config.get(p, [])
            L.append(k)
            inverted_config[p] = L
    return inverted_config


def detect_duplicate_parameters(
    inverted_config: dict,
    sub_config_name: str | None = None,
    ignore_keys: dict[str, Any] | None = None,
):
    if ignore_keys is None:
        ignore_keys = {'random': ('seed', 'samples')}

    duplicate_keys = []
    for p, L in inverted_config.items():
        if len(L) > 1:
            if 'random' in L and p in ignore_keys['random']:
                continue
            duplicate_keys.append((p, L))

    if len(duplicate_keys) > 0:
        if sub_config_name:
            raise ConfigError(
                f'Found duplicate keys in sub-config {sub_config_name}: '
                f'{duplicate_keys}'
            )
        else:
            raise ConfigError(f'Found duplicate keys: {duplicate_keys}')

    start_characters = {x[0] for x in inverted_config.keys()}
    buckets = {
        k: {x for x in inverted_config.keys() if x.startswith(k)}
        for k in start_characters
    }

    if sub_config_name:
        error_str = (
            f'Conflicting parameters in sub-config {sub_config_name}, most likely '
            'due to ambiguous use of dot-notation in the config dict. Found '
            "parameter '{p1}' in dot-notation starting with other parameter "
            "'{p2}', which is ambiguous."
        )
    else:
        error_str = (
            'Conflicting parameters, most likely '
            'due to ambiguous use of dot-notation in the config dict. Found '
            "parameter '{p1}' in dot-notation starting with other parameter "
            "'{p2}', which is ambiguous."
        )

    for k in buckets.keys():
        for p1, p2 in itertools.combinations(buckets[k], r=2):
            if p1.startswith(
                f'{p2}.'
            ):  # with "." after p2 to catch cases like "test" and "test1", which are valid.
                raise ConfigError(error_str.format(p1=p1, p2=p2))
            elif p2.startswith(f'{p1}.'):
                raise ConfigError(error_str.format(p1=p1, p2=p2))


def generate_configs(experiment_config, overwrite_params=None):
    """Generate parameter configurations based on an input configuration.

    Input is a nested configuration where on each level there can be 'fixed', 'grid', and 'random' parameters.

    In essence, we take the cartesian product of all the `grid` parameters and take random samples for the random
    parameters. The nested structure makes it possible to define different parameter spaces e.g. for different datasets.
    Parameter definitions lower in the hierarchy overwrite parameters defined closer to the root.

    For each leaf configuration we take the maximum of all num_samples values on the path since we need to have the same
    number of samples for each random parameter.

    For each configuration of the `grid` parameters we then create `num_samples` configurations of the random
    parameters, i.e. leading to `num_samples * len(grid_configurations)` configurations.

    See Also `examples/example_config.yaml` and the example below.

    Parameters
    ----------
    experiment_config: dict
        Dictionary that specifies the "search space" of parameters that will be enumerated. Should be
        parsed from a YAML file.
    overwrite_params: Optional[dict]
        Flat dictionary that overwrites configs. Resulting duplicates will be removed.

    Returns
    -------
    all_configs: list of dicts
        Contains the individual combinations of the parameters.


    """

    reserved, next_level = unpack_config(experiment_config)
    reserved = standardize_config(reserved)
    if not any([len(reserved.get(k, {})) > 0 for k in RESERVED_KEYS]):
        raise ConfigError(
            'No parameters defined under grid, fixed, or random in the config file.'
        )
    level_stack = [('', next_level)]
    config_levels = [reserved]
    final_configs = []

    detect_duplicate_parameters(invert_config(reserved), None)

    while len(level_stack) > 0:
        current_sub_name, sub_vals = level_stack.pop(0)
        sub_config, sub_levels = unpack_config(sub_vals)
        if current_sub_name != '' and not any(
            [len(sub_config.get(k, {})) > 0 for k in RESERVED_KEYS]
        ):
            raise ConfigError(
                f'No parameters defined under grid, fixed, or random in sub-config {current_sub_name}.'
            )
        sub_config = standardize_config(sub_config)
        config_above = config_levels.pop(0)

        inverted_sub_config = invert_config(sub_config)
        detect_duplicate_parameters(inverted_sub_config, current_sub_name)

        inverted_config_above = invert_config(config_above)
        redefined_parameters = set(inverted_sub_config.keys()).intersection(
            set(inverted_config_above.keys())
        )

        if len(redefined_parameters) > 0:
            logging.info(
                f"Parameters {redefined_parameters} are redefined in sub-config '{current_sub_name}'.\n"
                'Definitions in sub-configs override more general ones.'
            )
            config_above = copy.deepcopy(config_above)
            for p in redefined_parameters:
                sections = inverted_config_above[p]
                for s in sections:
                    del config_above[s][p]

        config = merge_dicts(config_above, sub_config)

        if len(sub_levels) == 0:
            final_configs.append((current_sub_name, config))

        for sub_name, sub_vals in sub_levels.items():
            new_sub_name = (
                f'{current_sub_name}.{sub_name}' if current_sub_name != '' else sub_name
            )
            level_stack.append((new_sub_name, sub_vals))
            config_levels.append(config)

    all_configs = []
    for subconfig_name, conf in final_configs:
        conf = standardize_config(conf)
        random_params = conf.get('random', {})
        fixed_params = flatten(conf.get('fixed', {}))
        grid_params = conf.get('grid', {})

        grids = [generate_grid(v, parent_key=k) for k, v in grid_params.items()]
        grid_configs = dict([sub for item in grids for sub in item])
        grouped_configs = zipped_dict(grid_configs)
        grid_product = list(cartesian_product_zipped_dict(grouped_configs))

        with_fixed = [{**d, **fixed_params} for d in grid_product]
        if len(random_params) > 0:
            num_samples = random_params['samples']
            root_seed = random_params.get('seed', None)
            random_sampled = sample_random_configs(
                flatten(random_params), seed=root_seed, samples=num_samples
            )
            with_random = [
                {**grid, **random} for grid in with_fixed for random in random_sampled
            ]
        else:
            with_random = with_fixed
        all_configs.extend(with_random)

    # Cast NumPy integers to normal integers since PyMongo doesn't like them
    all_configs = [
        {
            k: int(v)
            if isinstance(v, numbers.Integral) and not isinstance(v, bool)
            else v
            for k, v in config.items()
        }
        for config in all_configs
    ]

    if overwrite_params is not None:
        all_configs = [merge_dicts(config, overwrite_params) for config in all_configs]
        base_length = len(all_configs)
        # We use a dictionary instead a set because dictionary keys are ordered as of Python 3
        all_configs = list({Hashabledict(**config): None for config in all_configs})
        new_length = len(all_configs)
        if base_length != new_length:
            diff = base_length - new_length
            logging.warning(
                f'Parameter overwrite caused {diff} identical configs. Duplicates were removed.'
            )

    all_configs = [unflatten(conf) for conf in all_configs]
    return all_configs


def generate_named_config(named_config_dict: dict) -> list[str]:
    """Generates a sequence of named configs that is resolved by sacred in-order

    Parameters
    ----------
    named_config_dict : Dict
        Flattened configuration before parsing the named configurations.

    Returns
    -------
    List[str]
        A sequence of named configuration in the order that is defined by the `named_config_dict` input
    """
    # Parse named config names and priorities
    names, priorities = {}, {}
    for k, v in named_config_dict.items():
        if k.startswith(SETTINGS.NAMED_CONFIG.PREFIX):
            if isinstance(v, str):
                v = dict(name=str(v))
            if not isinstance(v, Dict):
                raise ConfigError(
                    f'Named configs must be given as '
                    f'{SETTINGS.NAMED_CONFIG.PREFIX}{"{identifier}"}: str | '
                    '{"name": str, "priority": int}'
                )
            for attribute, value in v.items():
                if attribute == SETTINGS.NAMED_CONFIG.KEY_NAME:
                    if not isinstance(value, str):
                        raise ConfigError(
                            f'Named config names should be strings, not {value} ({value.__class__})'
                        )
                    names[k] = value
                elif attribute == SETTINGS.NAMED_CONFIG.KEY_PRIORITY:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        raise ConfigError(
                            f'Named config priorities should be non-negative integers, not {value} ({value.__class__})'
                        )
                    priorities[k] = value
                else:
                    raise ConfigError(
                        f'Named configs only have the attributes {[SETTINGS.NAMED_CONFIG.KEY_NAME, SETTINGS.NAMED_CONFIG.KEY_PRIORITY]}'
                    )
    for idx in priorities:
        if idx not in names:
            raise ConfigError(
                f'Defined a priority but not a name for named config {idx}'
            )
    return [
        names[idx]
        for idx in sorted(
            names, key=lambda idx: (priorities.get(idx, float('inf')), names[idx])
        )
    ]


def generate_named_configs(configs: list[dict]) -> tuple[list[dict], list[list[str]]]:
    """From experiment configurations, generates both the config updates as well as the named configs in the order specified.

    Parameters
    ----------
    configs : List[Dict]
        Input configurations.

    Returns
    -------
    List[Dict]
        For each input configuration, the output configuration that does not contain named configuration specifiers anymore.

    List[List[str]]]
        For each input configuration, the sequence of named configurations in the order specified.
    """
    result_configs, result_named_configs = [], []
    for config in configs:
        result_configs.append(
            {
                k: v
                for k, v in config.items()
                if not k.startswith(SETTINGS.NAMED_CONFIG.PREFIX)
            }
        )
        result_named_configs.append(generate_named_config(config))
    return result_configs, result_named_configs


@functools.lru_cache
def load_config_dict(cfg_name: str):
    """
    Wrapper around sacred internal function to load a configuration file.
    This wrapper is cached to avoid loading the same file multiple times.

    Parameters
    ----------
    cfg_name : str
        Path to the configuration file.

    Returns
    -------
    ConfigDict
        The configuration dictionary.
    """
    from sacred.config.config_dict import ConfigDict
    from sacred.config.config_files import load_config_file

    return ConfigDict(load_config_file(cfg_name))


_SCAFFOLD_KEYS = (
    'config_updates',
    'named_configs_to_use',
    'config',
    'fallback',
    'presets',
    'fixture',
    'logger',
    'seed',
    'rnd',
    'config_mods',
    'summaries',
)


def _get_scaffold_state(scaffolding):
    return {
        k2: {k: getattr(scaffold, k) for k in _SCAFFOLD_KEYS}
        for k2, scaffold in scaffolding.items()
    }


def _set_scaffold_state(scaffolding, state):
    from copy import copy

    for k2, scaffold in scaffolding.items():
        for k, v in state[k2].items():
            setattr(scaffold, k, copy(v))


def _sacred_create_configs(
    exp: sacred.Experiment,
    configs: list[dict],
    named_configs: Sequence[Sequence[str]] | None = None,
) -> list[dict]:
    """Creates configs from an experiment and update values. This is done by re-implementing sacreds `sacred.initialize.create_run`
    method. Doing this is significantly faster, but it can be out-of-sync with sacred's current implementation.

    Parameters
    ----------
    exp : sacred.Experiment
        The sacred experiment to create configs for
    configs : List[Dict]
        Configuration updates for each experiment
    named_configs : Optional[List[Tuple[str]]], optional
        Named configs for each experiment, by default ()

    Returns
    -------
    Dict
        The updated configurations containing all values derived from the experiment.
    """
    from copy import deepcopy

    from sacred.config.utils import undogmatize
    from sacred.initialize import (
        Scaffold,
        create_scaffolding,
        distribute_config_updates,
        distribute_presets,
        gather_ingredients_topological,
        get_configuration,
        get_scaffolding_and_config_name,
    )
    from sacred.utils import (
        convert_to_nested_dict,
        iterate_flattened,
        join_paths,
        recursive_update,
        set_by_dotted_path,
    )

    from seml.console import track

    def run_named_config(scaffold: Scaffold, cfg_name: str):
        # This version of sacred.initialize.Scaffold.run_named_config uses our
        # cached version of load_config_dict. This is necessary to avoid loading the same file multiple times.
        if os.path.isfile(cfg_name):
            nc = load_config_dict(cfg_name)
            cfg = nc(
                fixed=scaffold.get_config_updates_recursive(),
                preset=scaffold.presets,
                fallback=scaffold.fallback,
            )
            return undogmatize(cfg)
        return scaffold.run_named_config(cfg_name)

    configs_resolved = []
    if named_configs is None:
        named_configs = [()] * len(configs)
    sorted_ingredients = gather_ingredients_topological(exp)
    scaffolding = create_scaffolding(exp, sorted_ingredients)
    init_state = deepcopy(_get_scaffold_state(scaffolding))
    # get all split non-empty prefixes sorted from deepest to shallowest
    prefixes = sorted(
        [s.split('.') for s in scaffolding if s != ''],
        reverse=True,
        key=lambda p: len(p),
    )

    for config, named_config in track(
        list(zip(configs, named_configs)),
        description='Resolving configurations',
        disable=len(configs) < SETTINGS.CONFIG_RESOLUTION_PROGRESS_BAR_THRESHOLD,
    ):
        # The following code is adapted from sacred directly: This results in a significant speedup
        # as we only care about the config but not about creating runs, however it is more error prone
        # to changes to sacred
        _set_scaffold_state(scaffolding, init_state)
        # --------- configuration process -------------------
        # Phase 1: Config updates
        config_updates = convert_to_nested_dict(config)
        distribute_config_updates(prefixes, scaffolding, config_updates)

        # Phase 2: Named Configs
        for ncfg in named_config:
            scaff, cfg_name = get_scaffolding_and_config_name(ncfg, scaffolding)
            scaff.gather_fallbacks()
            ncfg_updates = run_named_config(scaff, cfg_name)
            distribute_presets(scaff.path, prefixes, scaffolding, ncfg_updates)
            for ncfg_key, value in iterate_flattened(ncfg_updates):
                set_by_dotted_path(
                    config_updates, join_paths(scaff.path, ncfg_key), value
                )

        distribute_config_updates(prefixes, scaffolding, config_updates)

        # Phase 3: Normal config scopes
        for scaffold in scaffolding.values():
            scaffold.gather_fallbacks()
            scaffold.set_up_config()

            # update global config
            config = get_configuration(scaffolding)
            # run config hooks
            config_hook_updates = scaffold.run_config_hooks(
                config, exp.default_command, None
            )
            recursive_update(scaffold.config, config_hook_updates)

        # Phase 4: finalize seeding
        for scaffold in reversed(list(scaffolding.values())):
            scaffold.set_up_seed()  # partially recursive

        config_resolved = get_configuration(scaffolding)
        configs_resolved.append(
            remove_keys_from_nested(config_resolved, config_get_exclude_keys(config))
        )
    return configs_resolved


def resolve_configs(
    executable: str,
    conda_env: str | None,
    configs: list[dict],
    named_configs: list[list[str]],
    working_dir: str,
) -> list[dict]:
    """Resolves configurations by adding keys that are only added when the experiment is run to the MongoDB

    Parameters
    ----------
    executable : str
        Path to the executable
    conda_env : str
        Which conda environment to use
    configs : List[Dict]
        All experiment configurations
    named_configs : List[str]
        For each experiment, the named configurations to use.
    working_dir : str
        Which working directory to use

    Returns
    -------
    List[Dict]
        Resolved configurations
    """
    import sacred

    from seml.experiment.experiment import Experiment

    exp_module = import_exe(executable, conda_env, working_dir)

    # Extract experiment from module
    exps = [
        v
        for k, v in exp_module.__dict__.items()
        if isinstance(v, (sacred.Experiment, Experiment))
    ]
    if len(exps) == 0:
        raise ExecutableError(
            f"Found no Sacred experiment. Something is wrong in '{executable}'."
        )
    elif len(exps) > 1:
        raise ExecutableError(
            f"Found more than 1 Sacred experiment in '{executable}'. "
            f"Can't resolve configs."
        )
    exp = exps[0]
    if not isinstance(exp, Experiment):
        logging.warning(
            'The use of sacred.Experiment is deprecated. Please use seml.Experiment instead.\n'
            'seml.Experiment already includes typical MongoDB observer and logging setups.\n'
            'Please familiar yourself with the new API and adjust your code accordingly.\n'
            'See https://github.com/TUM-DAML/seml/blob/master/examples/example_experiment.py'
        )
    with working_directory(working_dir):
        return _sacred_create_configs(exp, configs, named_configs)


def check_config(
    executable: str, conda_env: str | None, configs: list[dict], working_dir: str
):
    """Check if the given configs are consistent with the Sacred experiment in the given executable.

    Parameters
    ----------
    executable: str
        The Python file containing the experiment.
    conda_env: str
        The experiment's Anaconda environment.
    configs: List[Dict]
        Contains the parameter configurations.
    working_dir : str
        The current working directory.
    """
    import sacred
    import sacred.initialize
    import sacred.utils

    exp_module = import_exe(executable, conda_env, working_dir)

    # Extract experiment from module
    exps = [
        v for k, v in exp_module.__dict__.items() if isinstance(v, sacred.Experiment)
    ]
    if len(exps) == 0:
        raise ExecutableError(
            f"Found no Sacred experiment. Something is wrong in '{executable}'."
        )
    elif len(exps) > 1:
        raise ExecutableError(
            f"Found more than 1 Sacred experiment in '{executable}'. "
            f"Can't check parameter configs. Disable via --no-sanity-check."
        )
    exp = exps[0]

    empty_run = sacred.initialize.create_run(
        exp, exp.default_command, config_updates=None, named_configs=()
    )

    captured_args = {
        sacred.utils.join_paths(cf.prefix, n)
        for cf in exp.captured_functions
        for n in cf.signature.arguments
    }

    config_keys_empty_run = set(flatten(empty_run.config).keys())

    for config in configs:
        config_flat = flatten(config)
        config_keys_added = set(config_flat.keys()).difference(config_keys_empty_run)

        # Check for unused arguments
        for conf in sorted(config_keys_added):
            if not (set(sacred.utils.iter_prefixes(conf)) & captured_args):
                raise sacred.utils.ConfigAddedError(
                    conf,
                    config={
                        k: v for k, v in config_flat.items() if k in config_keys_added
                    },
                )

        # Check for missing arguments
        options = empty_run.config.copy()
        options.update(config)
        options.update({k: None for k in sacred.utils.ConfigAddedError.SPECIAL_ARGS})
        try:
            empty_run.main_function.signature.construct_arguments(
                (), {}, options, False
            )
        except sacred.utils.MissingConfigError as e:
            logging.error(str(e))
            exit(1)


def restore(flat):
    """
    Restore more complex data that Python's json can't handle (e.g. Numpy arrays).
    Copied from sacred.serializer for performance reasons.
    """
    import json

    import jsonpickle

    return jsonpickle.decode(json.dumps(flat), keys=True)


def _convert_value(value):
    """
    Parse string as python literal if possible and fallback to string.
    Copied from sacred.arg_parser for performance reasons.
    """

    try:
        return restore(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value


def convert_values(val: Any):
    if isinstance(val, dict):
        for key, inner_val in val.items():
            val[key] = convert_values(inner_val)
    elif isinstance(val, list):
        for i, inner_val in enumerate(val):
            val[i] = convert_values(inner_val)
    elif isinstance(val, str):
        return _convert_value(val)
    return val


def read_config(config_path: str | Path):
    import yaml

    from seml import __version__
    from seml.utils.yaml import YamlUniqueLoader

    with open(config_path) as conf:
        config_dict = cast(
            SemlExperimentFile, convert_values(yaml.load(conf, Loader=YamlUniqueLoader))
        )

    if 'seml' not in config_dict:
        raise ConfigError("Please specify a 'seml' dictionary.")

    seml_conf = config_dict['seml']

    for k in seml_conf.keys():
        if k not in SETTINGS.VALID_SEML_CONFIG_VALUES:
            raise ConfigError(f'{k} is not a valid value in the `seml` config block.')

    if SETTINGS.SEML_CONFIG_VALUE_VERSION in seml_conf:
        raise ConfigError(
            f'Using {SETTINGS.SEML_CONFIG_VALUE_VERSION} in the `seml` config block is prohibited.'
        )

    version_array = [(int(x) if x.isdecimal() else x) for x in __version__.split('.')]
    executable, working_dir, output_dir, use_uploaded_sources = (
        determine_executable_and_working_dir(config_path, seml_conf)
    )
    seml = to_typeddict(seml_conf, SemlDocBase)
    seml.update(executable=executable)
    seml = SemlConfig(
        **seml,
        version=version_array,
        working_dir=working_dir,
        use_uploaded_sources=use_uploaded_sources,
        env=dict(os.environ),
    )
    if output_dir is not None:
        seml['output_dir'] = output_dir

    # Get list of slurm configs
    slurm_list: list[SlurmConfig] = config_dict.get('slurm', [])

    # Check for deprecated `slurm` dictionary
    if isinstance(slurm_list, dict):
        warnings.warn('`slurm` is expected to be a list of slurm configurations.')
        slurm_list = [cast(SlurmConfig, slurm_list)]

    if slurm_list is None:
        slurm_list: list[SlurmConfig] = []

    # Sanity check
    for slurm_conf in slurm_list:
        for k in slurm_conf.keys():
            if k not in SETTINGS.VALID_SLURM_CONFIG_VALUES:
                raise ConfigError(
                    f'{k} is not a valid value in the `slurm` config block.'
                )
        if slurm_conf.get('sbatch_options', None) is None:
            slurm_conf['sbatch_options'] = {}

    # If we have no config, we should add one
    if len(slurm_list) == 0:
        slurm_list.append(SlurmConfig(experiments_per_job=1, sbatch_options={}))

    # Remove unnecessary keys from config_dict
    config_dict = drop_typeddict_difference(
        config_dict, SemlExperimentFile, ExperimentConfig
    )
    return seml, slurm_list, config_dict


def determine_executable_and_working_dir(
    config_path: str | Path, seml_dict: SemlFileConfig
):
    """
    Determine the working directory of the project and chdir into the working directory.
    Parameters
    ----------
    config_path: Path to the config file
    seml_dict: SEML config dictionary

    Returns
    -------
    None
    """
    config_dir = str(Path(config_path).expanduser().resolve().parent)
    working_dir = config_dir
    if 'executable' not in seml_dict:
        raise ConfigError('Please specify an executable path for the experiment.')
    executable = seml_dict['executable']
    with working_directory(working_dir):
        executable_relative_to_config = os.path.exists(executable)
    executable_relative_to_project_root = False
    if 'project_root_dir' in seml_dict:
        with working_directory(config_dir):
            working_dir = str(
                Path(seml_dict['project_root_dir']).expanduser().resolve()
            )
        use_uploaded_sources = True
        with working_directory(working_dir):  # use project root as base dir from now on
            executable_relative_to_project_root = os.path.exists(executable)
        del seml_dict['project_root_dir']  # from now on we use only the working dir
    else:
        use_uploaded_sources = False
        logging.warning(
            "'project_root_dir' not defined in seml config. Source files will not be saved in MongoDB."
        )
    if not (executable_relative_to_config or executable_relative_to_project_root):
        raise ExecutableError('Could not find the executable.')
    with working_directory(working_dir):
        executable = str(Path(executable).expanduser().resolve())
        if executable_relative_to_project_root:
            executable = str(Path(executable).relative_to(working_dir))
        else:
            executable = str(Path(executable).relative_to(config_dir))

        if 'output_dir' in seml_dict:
            output_dir = seml_dict['output_dir']
        else:
            output_dir = None
    return executable, working_dir, output_dir, use_uploaded_sources


def remove_prepended_dashes(param_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a new dictionary where all keys that start with a dash are stripped of the dash.

    Parameters
    ----------
    param_dict : Dict[str, Any]
        The dictionary to remove the dashes from.

    Returns
    -------
    Dict[str, Any]
        The dictionary with the dashes removed.
    """
    new_dict = {}
    for k, v in param_dict.items():
        if k.startswith('--'):
            new_dict[k[2:]] = v
        elif k.startswith('-'):
            new_dict[k[1:]] = v
        else:
            new_dict[k] = v
    return new_dict


def config_get_exclude_keys(config_unresolved: dict | None = None) -> list[str]:
    """Gets the key that should be excluded from identifying a config. These should
    e.g. not be used in hashing

    Parameters
    ----------
    config_unresolved : Dict
        the configuration before resolution by sacred

    Returns
    -------
    List[str]
        keys that do not identify the config
    """
    exclude_keys = list(SETTINGS.CONFIG_EXCLUDE_KEYS)
    if config_unresolved is None:
        return exclude_keys
    if SETTINGS.CONFIG_KEY_SEED not in config_unresolved:
        # The seed will only be included (e.g. for hashing) if explicited in the unresolved configuration
        exclude_keys.append(SETTINGS.CONFIG_KEY_SEED)
    return exclude_keys


def create_starts_with_regex(*strings: str):
    """
    Creates a regex pattern that matches the start of a string with any of the given strings.

    Parameters
    ----------
    strings : List[str]
        The strings to match the start of.

    Returns
    -------
    re.Pattern
        The compiled regex pattern.
    """
    import re

    if len(strings) == 0:
        # Match not x and x -> always False
        return re.compile(r'^(?!x)x$')

    # Escape special characters in each string
    escaped_strings = [re.escape(s) for s in set(strings)]
    # Join the strings with '|' to create an OR pattern
    pattern = '|'.join(escaped_strings)
    # Add '^' to ensure the match is at the start of the string
    regex = f'^({pattern})'
    return re.compile(regex)


def requires_interpolation(
    document: Mapping[str, Any],
    allow_interpolation_keys: Iterable[str] = SETTINGS.ALLOW_INTERPOLATION_IN,
) -> bool:
    r"""
    Check if a document requires variable interpolation. This is done by checking if
    any value matches the regex: .*(?<!\\)\${.+}.*

    Parameters
    ----------
    document : Dict
        The document to check
    allow_interpolation_keys : List[str]
        All keys that should be permitted to do variable interpolation. Other keys are taken from the unresolved config.

    Returns
    -------
    bool
        True if the document requires variable interpolation
    """
    import re

    flat_dict = flatten(document)
    # Find a ${...} pattern that is not preceded by a backslash
    pattern = re.compile(r'.*(?<!\\)\${.+}.*')
    key_pattern = create_starts_with_regex(*allow_interpolation_keys)

    def check_interpolation(key, value):
        # These instructions are ordered by cost
        if not isinstance(value, str):
            return False
        if not pattern.match(value):
            return False
        return key_pattern.match(key)

    return any(map(check_interpolation, flat_dict.keys(), flat_dict.values()))


def escape_non_interpolated_dollars(
    document: Mapping[str, Any],
    allow_interpolation_keys: Iterable[str] = SETTINGS.ALLOW_INTERPOLATION_IN,
) -> dict[str, Any]:
    r"""
    Escapes all dollar signs that are not part of a variable interpolation.

    Parameters
    ----------
    document : Dict
        The document to escape.

    Returns
    -------
    Dict
        The escaped document
    """
    from seml.utils import unflatten

    flat_doc = flatten(document)
    key_pattern = create_starts_with_regex(*allow_interpolation_keys)
    for key, value in flat_doc.items():
        if isinstance(value, str) and not key_pattern.match(key):
            value = value.replace(r'${', r'\${')
            flat_doc[key] = value
    return unflatten(flat_doc)


T = TypeVar('T', bound=Mapping[str, Any])


def resolve_interpolations(
    document: T,
    allow_interpolation_keys: Iterable[str] = SETTINGS.ALLOW_INTERPOLATION_IN,
) -> T:
    """Resolves variable interpolation using `OmegaConf`

    Parameters
    ----------
    documents : Dict
        The document to resolve.
    allow_interpolations_in : List[str]
        All keys that should be permitted to do variable interpolation. Other keys are taken from the unresolved config.

    Returns
    -------
    Dict
        The resolved document
    """
    allow_interpolation_keys = set(allow_interpolation_keys)
    if not requires_interpolation(document, allow_interpolation_keys):
        return document

    from omegaconf import OmegaConf

    to_resolve_doc = escape_non_interpolated_dollars(document, allow_interpolation_keys)
    key_pattern = create_starts_with_regex(*allow_interpolation_keys)
    resolved = cast(
        T,
        OmegaConf.to_container(
            OmegaConf.create(to_resolve_doc, flags={'allow_objects': True}),
            resolve=True,
        ),
    )
    resolved_flat = {
        key: value for key, value in flatten(resolved).items() if key_pattern.match(key)
    }
    unresolved_flat = {
        key: value
        for key, value in flatten(document).items()
        if not key_pattern.match(key)
    }
    resolved_keys = set(resolved_flat.keys())
    unresolved_keys = set(unresolved_flat.keys())
    assert resolved_keys.isdisjoint(
        unresolved_keys
    ), f'Overlap between unresolved and resolved dicts: {resolved_keys.intersection(unresolved_keys)}'
    resolved = unflatten({**resolved_flat, **unresolved_flat})
    return cast(T, resolved)


def remove_duplicates_in_list(documents: Sequence[ExperimentDoc], use_hash: bool):
    """
    Returns a new list of ExperimentDoc where all elements are unique.

    Parameters
    ----------
    documents: Sequence[ExperimentDoc]
        The documents to filter.
    use_hash : bool
        Whether to use hashes (faster)

    Returns
    -------
    List[ExperimentDoc]
        List of unique documents.
    """
    if not use_hash:
        # slow duplicate detection without hashes
        unique_documents, unique_keys = [], set()
        for document in documents:
            key = Hashabledict(
                **remove_keys_from_nested(
                    document['config'],
                    config_get_exclude_keys(document['config_unresolved']),
                )
            )
            if key not in unique_keys:
                unique_documents.append(document)
                unique_keys.add(key)
        documents = unique_documents
    else:
        # fast duplicate detection using hashing.
        documents_dict = {document['config_hash']: document for document in documents}
        documents = list(documents_dict.values())
    return documents


def remove_duplicates_in_db(
    collection: Collection[ExperimentDoc],
    documents: Sequence[ExperimentDoc],
    use_hash: bool,
):
    """Check database collection for already present entries.

    Check the database collection for experiments that have the same configuration.
    Remove the corresponding entries from the input list of configurations to prevent
    re-running the experiments.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    documents: List[Dict]
        The documents to filter.
    use_hash : bool
        Whether to use hashes (faster)

    Returns
    -------
    filtered_configs: list of dicts
        No longer contains configurations that are already in the database collection.

    """
    if use_hash:
        hashes = [document['config_hash'] for document in documents]
        db_hashes = set(
            collection.find({'config_hash': {'$in': hashes}}).distinct('config_hash')
        )
        return [doc for doc in documents if doc['config_hash'] not in db_hashes]

    filtered_documents: list[ExperimentDoc] = []
    for document in documents:
        lookup_dict = flatten(
            {
                'config': remove_keys_from_nested(
                    document['config'], document['config_unresolved'].keys()
                )
            }
        )
        lookup_result = collection.find_one(unflatten(lookup_dict))
        if lookup_result is None:
            filtered_documents.append(document)
    return filtered_documents


def remove_duplicates(
    collection: Collection[ExperimentDoc] | None,
    documents: Sequence[ExperimentDoc],
    use_hash: bool = True,
):
    """
    Returns a new list of documents that do not contain duplicates in the database or within the input list.

    Parameters
    ----------
    collection: pymongo.collection.Collection
        The MongoDB collection containing the experiments.
    documents: Sequence[ExperimentDoc]
        The documents to filter.
    use_hash : bool
        Whether to use hashes (faster)

    Returns
    -------
    filtered_configs: list of ExperimentDoc
        No longer contains configurations that are already in the database collection.
    """
    if len(documents) == 0:
        return list(documents)
    n_total = len(documents)

    # First, check for duplicates withing the experiment configurations from the file.
    documents = remove_duplicates_in_list(documents, use_hash)
    n_unique = len(documents)
    if n_unique != n_total:
        logging.info(
            f'{n_total - n_unique} of {n_total} experiment{s_if(n_total)} were '
            f'duplicates. Adding only the {n_unique} unique configurations.'
        )
    # Now, check for duplicate configurations in the database.
    if collection is not None:
        documents = remove_duplicates_in_db(collection, documents, use_hash)
        n_unique_and_not_in_db = len(documents)
        if n_unique_and_not_in_db != n_unique:
            logging.info(
                f'{n_unique - n_unique_and_not_in_db} of {n_unique} '
                f'experiment{s_if(n_unique)} were already found in the database. They were not added again.'
            )
    return documents


def check_slurm_config(experiments_per_job: int, sbatch_options: SBatchOptions):
    if not (
        (
            sbatch_options.get('nodes', 1) == 1
            and sbatch_options.get('N', 1) == 1  # short for --nodes
            and sbatch_options.get('ntasks', 1) == 1
            and sbatch_options.get('n', 1) == 1  # short for --ntasks
            and sbatch_options.get('ntasks-per-node', 1) == 1
            and sbatch_options.get('ntasks-per-gpu', 1) == 1
            and sbatch_options.get('ntasks-per-socket', 1) == 1
        )
        or experiments_per_job == 1
    ):
        raise ConfigError(
            'Cannot run multiple experiments per job with multiple nodes or tasks per node.'
        )


def assemble_slurm_config_dict(experiment_slurm_config: SlurmConfig):
    """
    Realize inheritance for the slurm configuration, with the following relationship:
    Default -> Template -> Experiment

    Parameters
    ----------
    experiment_slurm_config: The slurm experiment configuration as returned by the function read_config

    Returns
    -------
    slurm_config

    """
    # Rename
    slurm_config = experiment_slurm_config
    # Assemble the Slurm config:
    # Basis config is the default config. This can be overridden by the sbatch_options_template.
    # And this in turn can be overridden by the sbatch config defined in the experiment .yaml file.
    slurm_config_base = copy.deepcopy(SETTINGS.SLURM_DEFAULT)

    # Check for and use sbatch options template
    sbatch_options_template = slurm_config.get('sbatch_options_template', None)
    if sbatch_options_template is not None:
        if sbatch_options_template not in SETTINGS.SBATCH_OPTIONS_TEMPLATES:
            raise ConfigError(
                f"sbatch options template '{sbatch_options_template}' not found in settings.py."
            )
        slurm_config_base['sbatch_options'] = merge_dicts(
            slurm_config_base['sbatch_options'],
            SETTINGS.SBATCH_OPTIONS_TEMPLATES[sbatch_options_template],
        )

    # Integrate experiment specific config
    slurm_config = merge_dicts(slurm_config_base, slurm_config)

    slurm_config['sbatch_options'] = cast(
        SBatchOptions,
        remove_prepended_dashes(cast(Dict[str, Any], slurm_config['sbatch_options'])),
    )

    # Check that ntasks and experiments_per_job are mutually exclusive
    sbatch_options = slurm_config['sbatch_options']
    check_slurm_config(slurm_config.get('experiments_per_job', 1), sbatch_options)
    return slurm_config
