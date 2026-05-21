from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

from seml.settings import SETTINGS

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

_STOP_COMMAND = 'stop'


def retried_and_locked_ssh_port_forward(
    retries_max: int = SETTINGS.SSH_FORWARD.RETRIES_MAX,
    retries_delay: int = SETTINGS.SSH_FORWARD.RETRIES_DELAY,
    lock_file: str = SETTINGS.SSH_FORWARD.LOCK_FILE,
    lock_timeout: int = SETTINGS.SSH_FORWARD.LOCK_TIMEOUT,
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
            logging.warning(f'Failed to aquire lock for ssh tunnel {lock_file}')
        except BaseSSHTunnelForwarderError as e:
            error = e
            logging.warning(f'Retry establishing ssh tunnel in {delay} s')
            # Jittered exponential retry
            time.sleep(delay)
            delay *= 2
            delay += random.uniform(0, 1)

    logging.error(f'Failed to establish ssh tunnel: {error}')
    exit(1)


def _remove_unix_socket(socket_address: str):
    with contextlib.suppress(FileNotFoundError, OSError):
        os.unlink(socket_address)


def _ssh_forward_process(connection: Connection, ssh_config: dict[str, Any]):
    """
    Establish an SSH tunnel in a separate process. The process periodically checks if the tunnel is still up and
    restarts it if it is not.

    Parameters
    ----------
    connection: Connection
        Connection carrying commands from the parent process.
    ssh_config: dict
        Configuration for the SSH tunnel.
    """
    server = retried_and_locked_ssh_port_forward(**ssh_config)
    # We need to bind to the same local addresses
    server._local_binds = server.local_bind_addresses
    connection.send((str(server.local_bind_host), int(server.local_bind_port)))

    while True:
        # check if we should end the process
        try:
            command = None
            if connection.poll(SETTINGS.SSH_FORWARD.HEALTH_CHECK_INTERVAL):
                command = connection.recv()

            if command == _STOP_COMMAND:
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
        except (EOFError, OSError):
            server.stop()
            break
        except Exception as e:
            logging.error(f'Error in SSH tunnel health check:\n{e}')
            server.restart()


def main(socket_address: str, authkey_hex: str):
    from multiprocessing.connection import Listener

    try:
        authkey = bytes.fromhex(authkey_hex)
    except ValueError:
        logging.error('Invalid SSH worker authkey payload.')
        return 1

    listener = Listener(socket_address, family='AF_UNIX', authkey=authkey)
    connection: Connection | None = None
    try:
        connection = listener.accept()
        ssh_config = connection.recv()
        if not isinstance(ssh_config, dict):
            logging.error('SSH worker expects the configuration payload to be a dict.')
            return 1
        _ssh_forward_process(connection, ssh_config)
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        _remove_unix_socket(socket_address)
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit(
            'Usage: python -m seml.utils.ssh_tunnel_worker --worker <socket_address> <authkey_hex>'
        )
    raise SystemExit(main(sys.argv[2], sys.argv[3]))
