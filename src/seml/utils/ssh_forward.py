from __future__ import annotations

import atexit
import contextlib
import logging
import os
import secrets
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any

from seml.document import ExperimentDoc
from seml.settings import SETTINGS
from seml.utils import Hashabledict, assert_package_installed

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

States = SETTINGS.STATES
_STOP_COMMAND = 'stop'


def _remove_unix_socket(socket_address: str):
    with contextlib.suppress(FileNotFoundError, OSError):
        os.unlink(socket_address)


def _connect_to_ssh_worker(socket_address: str, authkey: bytes, timeout: float):
    from multiprocessing.connection import Client

    deadline = time.monotonic() + max(timeout, 0)
    while True:
        try:
            return Client(socket_address, family='AF_UNIX', authkey=authkey)
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError('Failed to connect to SSH forwarding worker.')
            time.sleep(0.05)


def _start_ssh_forward_subprocess(connect_timeout: float):
    socket_address = f'/tmp/seml_ssh_forward_{uuid.uuid4().hex}.sock'
    authkey = secrets.token_bytes(16)  # avoid other processes connecting to the socket
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'seml.utils.ssh_tunnel_worker',
            '--worker',
            socket_address,
            authkey.hex(),
        ]
    )

    try:
        connection = _connect_to_ssh_worker(socket_address, authkey, connect_timeout)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
        _remove_unix_socket(socket_address)
        raise
    return proc, connection, socket_address


def _close_ssh_forward_subprocess(
    proc: subprocess.Popen[Any],
    connection: Connection | None,
    socket_address: str,
):
    try:
        if proc.poll() is None and connection is not None:
            try:
                connection.send(_STOP_COMMAND)
            except (BrokenPipeError, EOFError, OSError):
                pass
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
        _remove_unix_socket(socket_address)


# To establish only a single connection to a remote
_forwards: dict[Hashabledict, tuple[str, int]] = {}
_forward_processes: dict[
    Hashabledict, tuple[subprocess.Popen[Any], Connection, str]
] = {}


def _get_ssh_forward(ssh_config: dict[str, Any]):
    """
    Establishes an SSH tunnel in a separate process and returns the local address of the tunnel.
    If a connection to the remote host already exists, it is reused.

    Parameters
    ----------
    ssh_config: dict
        Configuration for the SSH tunnel.

    Returns
    -------
    local_address: tuple
        Local address of the SSH tunnel.
    try_close: Callable
        Function to close the SSH tunnel.
    """
    assert_package_installed(
        'sshtunnel',
        'Opening ssh tunnel requires `sshtunnel` (e.g. `pip install sshtunnel`)',
    )
    assert_package_installed(
        'filelock',
        'Opening ssh tunnel requires `filelock` (e.g. `pip install filelock`)',
    )

    global _forwards, _forward_processes

    ssh_config = Hashabledict(ssh_config)
    if ssh_config not in _forwards:
        # Compute the maximum time we should wait
        retries_max = ssh_config.get('retries_max', SETTINGS.SSH_FORWARD.RETRIES_MAX)
        retries_delay = ssh_config.get(
            'retries_delay', SETTINGS.SSH_FORWARD.RETRIES_DELAY
        )
        max_delay = 2 ** (retries_max + 1) * retries_delay
        try:
            proc, connection, socket_address = _start_ssh_forward_subprocess(max_delay)
        except TimeoutError:
            logging.error('Failed to connect to SSH tunnel worker.')
            exit(1)

        # Send stop if we exit the program
        atexit.register(
            lambda: _close_ssh_forward_subprocess(proc, connection, socket_address)
        )

        try:
            connection.send(dict(ssh_config))
        except (BrokenPipeError, EOFError, OSError):
            logging.error('Failed to send SSH tunnel configuration to worker.')
            _close_ssh_forward_subprocess(proc, connection, socket_address)
            exit(1)

        if not connection.poll(max_delay):
            logging.error('Failed to establish SSH tunnel.')
            _close_ssh_forward_subprocess(proc, connection, socket_address)
            exit(1)

        try:
            host, port = connection.recv()
        except (EOFError, OSError, ValueError, TypeError) as e:
            logging.error(f'Failed to receive SSH tunnel worker startup output: {e}')
            _close_ssh_forward_subprocess(proc, connection, socket_address)
            exit(1)

        _forwards[ssh_config] = (str(host), int(port))
        _forward_processes[ssh_config] = (proc, connection, socket_address)
    return _forwards[ssh_config]


def get_forwarded_mongo_client(
    db_name: str, username: str, password: str, ssh_config: dict[str, Any], **kwargs
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
    import pymongo

    host, port = _get_ssh_forward(ssh_config)

    client = pymongo.MongoClient[ExperimentDoc](
        host,
        int(port),
        username=username,
        password=password,
        authSource=db_name,
        **kwargs,
    )
    return client
