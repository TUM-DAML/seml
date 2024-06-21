import logging

from seml.settings import SETTINGS


def prompt_ssh_forward():
    """
    Prompt the user for SSH Forward settings. The output format corresponds
    to the argument of sshtunnel.SSHTunnelForwarder.
    """
    from seml.console import prompt

    logging.info('Configuring SSH Forward settings.')
    ssh_host = prompt('SSH host')
    port = prompt('Port', default=22, type=int)
    username = prompt('User name')
    ssh_pkey = prompt('Path to SSH private key', default='~/.ssh/id_rsa')
    return dict(
        ssh_address_or_host=ssh_host,
        ssh_port=port,
        ssh_username=username,
        ssh_pkey=ssh_pkey,
    )


def mongodb_configure(setup_ssh_forward: bool = False):
    import yaml

    from seml.console import prompt

    if SETTINGS.DATABASE.MONGODB_CONFIG_PATH.exists() and not prompt(
        f'MongoDB configuration {SETTINGS.DATABASE.MONGODB_CONFIG_PATH} already exists and will be overwritten.\nContinue?',
        type=bool,
    ):
        return
    logging.info('Configuring MongoDB. Warning: Password will be stored in plain text.')
    host = prompt('MongoDB host')
    port = prompt('Port', default=27017, type=int)
    database = prompt('Database name')
    username = prompt('User name')
    password = prompt('Password', hide_input=True)
    file_path = SETTINGS.DATABASE.MONGODB_CONFIG_PATH
    config = dict(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )
    if setup_ssh_forward:
        config['ssh_config'] = prompt_ssh_forward()
    config_string = yaml.dump(config)
    logging.info(
        f"Saving the following configuration to {file_path}:\n"
        f"{config_string.replace(f'{password}', '********')}"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(config_string)


def configure(
    all: bool = False, mongodb: bool = False, setup_ssh_forward: bool = False
):
    configured_any = False
    if mongodb or all:
        mongodb_configure(setup_ssh_forward=setup_ssh_forward)
        configured_any = True
    if not configured_any:
        logging.info('Did not specify any configuration to configure')
