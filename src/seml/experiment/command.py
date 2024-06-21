import logging
import shlex
import urllib
from typing import Optional

from seml.experiment.config import generate_named_configs
from seml.experiment.config import (
    resolve_interpolations as resolve_config_interpolations,
)
from seml.settings import SETTINGS
from seml.utils.errors import ArgumentError, MongoDBError
from seml.utils.network import find_free_port


def _generate_debug_attach_url(ip_address, port):
    import json

    launch_config = {
        'type': 'debugpy',
        'request': 'attach',
        'connect': {'host': ip_address, 'port': port},
        'pathMappings': [{'localRoot': '${workspaceFolder}', 'remoteRoot': '.'}],
    }
    launch_config = urllib.parse.quote(json.dumps(launch_config))
    return f'vscode://fabiospampinato.vscode-debug-launcher/launch?args={launch_config}'


def get_environment_variables(gpus=None, cpus=None, environment_variables=None):
    if environment_variables is None:
        environment_variables = {}

    if gpus is not None:
        if isinstance(gpus, list):
            raise ArgumentError(
                'Received an input of type list to set CUDA_VISIBLE_DEVICES. '
                'Please pass a string for input "gpus", '
                'e.g. "1,2" if you want to use GPUs with IDs 1 and 2.'
            )
        environment_variables['CUDA_VISIBLE_DEVICES'] = str(gpus)
    if cpus is not None:
        environment_variables['OMP_NUM_THREADS'] = str(cpus)
    return environment_variables


def get_config_overrides(config):
    return ' '.join(map(shlex.quote, config))


def get_shell_command(interpreter, exe, config, env: Optional[dict] = None):
    config_overrides = get_config_overrides(config)

    if env is None or len(env) == 0:
        return f'{interpreter} {exe} with {config_overrides}'
    else:
        env_overrides = ' '.join(
            f'{key}={shlex.quote(val)}' for key, val in env.items()
        )

        return f'{env_overrides} {interpreter} {exe} with {config_overrides}'


def value_to_string(value, use_json=False):
    from seml.utils.json import PythonEncoder

    # We need the json encoding for vscode due to https://github.com/microsoft/vscode/issues/91578
    # Once this bug has been fixed we should only rely on `repr` and remove this code.
    if use_json:
        return PythonEncoder().encode(value)
    else:
        return repr(value)


def get_command_from_exp(
    exp,
    db_collection_name,
    verbose=False,
    unobserved=False,
    post_mortem=False,
    debug=False,
    debug_server=False,
    print_info=True,
    use_json=False,
    unresolved=False,
    resolve_interpolations: bool = True,
):
    from seml.console import console

    if 'executable' not in exp['seml']:
        raise MongoDBError(
            f"No executable found for experiment {exp['_id']}. Aborting."
        )
    exe = exp['seml']['executable']

    if unresolved:
        config_unresolved = exp.get('config_unresolved', exp['config'])
        config, named_configs = tuple(
            zip(*generate_named_configs([config_unresolved]))
        )[0]
        # Variable interpolation in unresolved and named configs

        if resolve_interpolations:
            import uuid

            key_named_configs = str(uuid.uuid4())
            interpolated = resolve_config_interpolations(
                {
                    **exp,
                    'config_unresolved': config_unresolved,
                    key_named_configs: named_configs,
                },
                allow_interpolation_keys=list(SETTINGS.ALLOW_INTERPOLATION_IN)
                + ['config_unresolved', key_named_configs],
            )

            config = {
                k: v
                for k, v in interpolated['config_unresolved'].items()
                if not k.startswith(SETTINGS.NAMED_CONFIG.PREFIX)
            }
            named_configs = interpolated[key_named_configs]
        else:
            config = {
                k: v
                for k, v in config_unresolved.items()
                if not k.startswith(SETTINGS.NAMED_CONFIG.PREFIX)
            }
    else:
        assert (
            resolve_interpolations
        ), 'In resolved configs, interpolations are automatically resolved'
        config = exp['config']
        named_configs = []

    config['db_collection'] = db_collection_name
    if not unobserved:
        config['overwrite'] = exp['_id']

    # We encode values with `repr` such that we can decode them with `eval`. While `shlex.quote`
    # may cause messy commands with lots of single quotes JSON doesn't match Python 1:1, e.g.,
    # boolean values are lower case in JSON (true, false) but start with capital letters in Python.
    config_strings = [
        f'{key}={value_to_string(val, use_json)}' for key, val in config.items()
    ]
    config_strings += named_configs

    # TODO (?): Variable interpolation for unresolved CLI calls

    if not verbose:
        config_strings.append('--force')
    if unobserved:
        config_strings.append('--unobserved')
    if post_mortem:
        config_strings.append('--pdb')
    if debug:
        config_strings.append('--debug')

    if debug_server:
        ip_address, port = find_free_port()
        if print_info:
            logging.info(
                f"Starting debug server with IP '{ip_address}' and port '{port}'. "
                f'Experiment will wait for a debug client to attach.'
            )
            attach_link = _generate_debug_attach_url(ip_address, port)

            logging.info(
                "If you are using VSCode, you can use the 'Debug Launcher' extension to attach:"
            )
            console.out(attach_link)

        interpreter = (
            f'python -m debugpy --listen {ip_address}:{port} --wait-for-client'
        )
    else:
        interpreter = 'python'

    return interpreter, exe, config_strings
