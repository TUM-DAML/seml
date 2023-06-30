import logging

from seml.settings import SETTINGS
from seml.typer import prompt


def mongodb_configure(): 
    if SETTINGS.DATABASE.MONGODB_CONFIG_PATH.exists() and not prompt(
        f'MongoDB configuration {SETTINGS.DATABASE.MONGODB_CONFIG_PATH} already exists and will be overwritten.\nContinue?',
        type=bool
    ):
        return
    logging.info('Configuring MongoDB. Warning: Password will be stored in plain text.')
    host = prompt("MongoDB host")
    port = prompt("Port", default="27017")
    database = prompt("Database name")
    username = prompt("User name")
    password = prompt("Password", hide_input=True)
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


def configure(all: bool=False, mongodb: bool=False):
    configured_any = False
    if mongodb or all:
        mongodb_configure()
        configured_any = True
    if not configured_any:
        logging.info('Did not specify any configuration to configure')
