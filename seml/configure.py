
from pathlib import Path
from typing import Optional, Iterable
import os
import logging
import click

from seml.settings import SETTINGS
from seml.errors import ArgumentError
import getpass

def get_input(field_name: Optional[str]=None, num_trials=3, password: bool=False,
                       choices: Optional[Iterable[str]]=None, default=None, nonempty: bool=True,
                       existing_path: bool=False,
                       prompt: Optional[str]=None) -> str:
    get_input = getpass if password else input
    for _ in range(num_trials):
        prompt = prompt or f"Please input the {field_name}"
        if choices is not None and len(choices) > 0:
            prompt += f', (one of [{", ".join(map(str, choices))}])'
        if default:
            prompt += f", default={default}"
        prompt += ": "
        field = get_input(prompt)
        if not field:
            field = default
        if nonempty and (field is None or len(field) == 0):
            logging.error(f'{field_name} was empty.')
            continue
        if choices is not None and field not in choices:
            logging.error(f'{field_name} must be one of {choices}')
            continue
        if existing_path and not Path(field).expanduser().resolve().exists():
            logging.error(f'Path {Path(field)} does not exist.')
            continue
        return field
    else:
        raise ArgumentError(f"Did not receive an input for {num_trials} times. Aborting.")

ARGCOMPLETE_GLOBAL = 'global'
ARGCOMPLETE_USER = 'user'
ARGCOMPLETE_PATH = 'path'

def argcomplete_configure():
    """ Initializes argument completion. """
    logging.info('Configuring argument completion.')
    mode = get_input(prompt='Enter how argument completion should be registered', 
                     choices=[ARGCOMPLETE_USER, ARGCOMPLETE_GLOBAL, ARGCOMPLETE_PATH], default=ARGCOMPLETE_USER)
    if mode == ARGCOMPLETE_GLOBAL:
        cmd = 'activate-global-python-argcomplete'
    elif mode == ARGCOMPLETE_USER:
        cmd = 'activate-global-python-argcomplete --user'
    elif mode == ARGCOMPLETE_PATH:
        path = get_input(prompt='Enter the shell completion modules directory to install to', existing_path=True)
        path = str(Path(path).expanduser().resolve())
        cmd = f'activate-global-python-argcomplete --dest {path}'
    else:
        raise ValueError(mode)
    logging.info(f'Running {cmd}')
    os.system(cmd)
        
def mongodb_configure():
    if SETTINGS.DATABASE.MONGODB_CONFIG_PATH.exists() and not click.confirm(
        f'MongoDB configuration {SETTINGS.DATABASE.MONGODB_CONFIG_PATH} already exists and will be overwritten. Continue?',
        default=False
    ):
        return
    logging.info('Configuring MongoDB. Warning: Password will be stored in plain text.')
    host = get_input("MongoDB host")
    port = input('Port (default: 27017):')
    port = "27017" if port == "" else port
    database = get_input("database name")
    username = get_input("user name")
    password = get_input("password", password=True)
    file_path = SETTINGS.DATABASE.MONGODB_CONFIG_PATH
    config_string = (f'username: {username}\n'
                     f'password: {password}\n'
                     f'port: {port}\n'
                     f'database: {database}\n'
                     f'host: {host}')
    logging.info(f"Saving the following configuration to {file_path}:\n"
                 f"{config_string.replace(f'password: {password}', 'password: ********')}"
                 )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(config_string)
        
def configure(all: bool=False, mongodb: bool=False, argcomplete: bool=False):
    configured_any = False
    if mongodb or all:
        mongodb_configure()
        configured_any = True
    if argcomplete or all:
        argcomplete_configure()
        configured_any = True
    if not configured_any:
        logging.info('Did not specify any configuration to configure')
    