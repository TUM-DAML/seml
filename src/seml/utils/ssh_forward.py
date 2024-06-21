import atexit
import logging
import time
from typing import Any, Dict

from seml.settings import SETTINGS
from seml.utils import assert_package_installed

States = SETTINGS.STATES


def retried_and_locked_ssh_port_forward(
    retries_max=SETTINGS.SSH_FORWARD.RETRIES_MAX,
    retries_delay=SETTINGS.SSH_FORWARD.RETRIES_DELAY,
    lock_file=SETTINGS.SSH_FORWARD.LOCK_FILE,
    lock_timeout=SETTINGS.SSH_FORWARD.LOCK_TIMEOUT,
    **ssh_config,
):
    """
    Attempt to establish an SSH tunnel with retries and a lock file to avoid parallel tunnel establishment.

    Parameters
    ----------
    retries_max: int
        Maximum number of retries to establish the tunnel.
    retries_delay: float
        Initial delay for exponential backoff.
    lock_file: str
        Path to the lock file.
    lock_timeout: int
        Timeout for acquiring the lock.
    ssh_config: dict
        Configuration for the SSH tunnel.

    Returns
    -------
    server: SSHTunnelForwarder
        The SSH tunnel server.
    """
    import random

    from filelock import FileLock, Timeout
    from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder

    delay = retries_delay
    error = None
    # disable SSH forward messages
    logging.getLogger('paramiko.transport').disabled = True
    for _ in range(retries_max):
        try:
            lock = FileLock(lock_file, mode=0o666, timeout=lock_timeout)
            with lock:
                server = SSHTunnelForwarder(**ssh_config)
                server.start()
                if not server.tunnel_is_up[server.local_bind_address]:
                    raise BaseSSHTunnelForwarderError()
                return server
        except Timeout as e:
            error = e
            logging.warn(f'Failed to aquire lock for ssh tunnel {lock_file}')
        except BaseSSHTunnelForwarderError as e:
            error = e
            logging.warn(f'Retry establishing ssh tunnel in {delay} s')
            # Jittered exponential retry
            time.sleep(delay)
            delay *= 2
            delay += random.uniform(0, 1)

    if error:
        logging.error(f'Failed to establish ssh tunnel: {error}')
        exit(1)


def _ssh_forward_process(pipe, ssh_config: Dict[str, Any]):
    """
    Establish an SSH tunnel in a separate process. The process periodically checks if the tunnel is still up and
    restarts it if it is not.

    Parameters
    ----------
    pipe: multiprocessing.communication.Connection
        Pipe to communicate with the main process.
    ssh_config: dict
        Configuration for the SSH tunnel.
    """
    server = retried_and_locked_ssh_port_forward(**ssh_config)
    # We need to bind to the same local addresses
    server._local_binds = server.local_bind_addresses
    pipe.send((server.local_bind_host, server.local_bind_port))
    while True:
        # check if we should end the process
        try:
            if pipe.poll(SETTINGS.SSH_FORWARD.HEALTH_CHECK_INTERVAL):
                if pipe.closed or pipe.recv() == 'stop':
                    server.stop()
                    break

            # Check for tunnel health
            server.check_tunnels()
            if not server.tunnel_is_up[server.local_bind_address]:
                logging.warning('SSH tunnel was closed unexpectedly. Restarting.')
                server.restart()
        except KeyboardInterrupt:
            server.stop()
            break
        except EOFError:
            server.stop()
            break
        except Exception as e:
            logging.error(f'Error in SSH tunnel health check:\n{e}')
            server.restart()
    pipe.close()


# We want to reuse the same multiprocessing context for all SSH tunnels
_mp_context = None


def get_forwarded_mongo_client(
    db_name, username, password, ssh_config: Dict[str, Any], **kwargs
):
    """
    Establish an SSH tunnel and return a forwarded MongoDB client.
    The SSH tunnel is established in a separate process to enable continuously checking for its health.

    Parameters
    ----------
    db_name: str
        Name of the database.
    username: str
        Username for the database.
    password: str
        Password for the database.
    ssh_config: dict
        Configuration for the SSH tunnel.
    kwargs: dict
        Additional arguments for the MongoDB client.

    Returns
    -------
    client: pymongo.MongoClient
        Forwarded MongoDB client.
    """
    import multiprocessing as mp

    import pymongo

    global _mp_context

    assert_package_installed(
        'sshtunnel',
        'Opening ssh tunnel requires `sshtunnel` (e.g. `pip install sshtunnel`)',
    )
    assert_package_installed(
        'filelock',
        'Opening ssh tunnel requires `filelock` (e.g. `pip install filelock`)',
    )

    if _mp_context is None:
        _mp_context = mp.get_context('forkserver')
    main_pipe, forward_pipe = _mp_context.Pipe(True)
    proc = _mp_context.Process(
        target=_ssh_forward_process, args=(forward_pipe, ssh_config)
    )
    proc.start()

    def try_close():
        try:
            if not main_pipe.closed:
                main_pipe.send('stop')
                main_pipe.close()
        finally:
            pass

    class ForwardedMongoClient(pymongo.MongoClient):
        def __del__(self):
            try_close()

    # Send stop if we exit the program
    atexit.register(try_close)

    # Compute the maximum time we should wait
    retries_max = ssh_config.get('retries_max', SETTINGS.SSH_FORWARD.RETRIES_MAX)
    retries_delay = ssh_config.get('retries_delay', SETTINGS.SSH_FORWARD.RETRIES_DELAY)
    max_delay = 2 ** (retries_max + 1) * retries_delay

    # check if the forward process has been established correctly
    if main_pipe.poll(max_delay):
        host, port = main_pipe.recv()

        client = ForwardedMongoClient(
            host,
            int(port),
            username=username,
            password=password,
            authSource=db_name,
            **kwargs,
        )
        return client
    else:
        logging.error('Failed to establish SSH tunnel.')
        exit(1)
