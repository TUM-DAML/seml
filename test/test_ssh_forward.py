from __future__ import annotations

import unittest
from typing import Any, cast
from unittest import mock

from seml.utils import ssh_forward, ssh_tunnel_worker


class FakeServer:
    def __init__(self):
        self.local_bind_addresses = [("127.0.0.1", 27017)]
        self.local_bind_address = ("127.0.0.1", 27017)
        self.local_bind_host = "127.0.0.1"
        self.local_bind_port = 27017
        self.tunnel_is_up = {self.local_bind_address: True}
        self.stopped = False
        self.restarted = False

    def check_tunnels(self):
        return None

    def restart(self):
        self.restarted = True

    def stop(self):
        self.stopped = True


class FakeConnection:
    def __init__(self, recv_values=None, poll_values=None):
        self._recv_values = list(recv_values or [])
        self._poll_values = list(poll_values or [])
        self.sent: list[object] = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def recv(self):
        if not self._recv_values:
            raise EOFError("No queued value to receive.")
        return self._recv_values.pop(0)

    def poll(self, _timeout):
        if self._poll_values:
            return self._poll_values.pop(0)
        return False

    def close(self):
        self.closed = True


class FakeWorkerProcess:
    def __init__(self):
        self._returncode = None
        self.terminated = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = 0


class TestSSHForwardWorker(unittest.TestCase):
    def test_ssh_forward_process_sends_startup_and_stops(self):
        fake_server = FakeServer()
        fake_connection = FakeConnection(recv_values=["stop"], poll_values=[True])

        with mock.patch.object(
            ssh_tunnel_worker,
            "retried_and_locked_ssh_port_forward",
            return_value=fake_server,
        ) as start_tunnel:
            ssh_tunnel_worker._ssh_forward_process(
                cast(Any, fake_connection),
                {"ssh_address_or_host": "jump-host.example"},
            )

        start_tunnel.assert_called_once_with(ssh_address_or_host="jump-host.example")
        self.assertEqual(fake_connection.sent, [("127.0.0.1", 27017)])
        self.assertTrue(fake_server.stopped)

    def test_worker_main_accepts_connection_and_runs_process(self):
        fake_connection = mock.Mock()
        fake_connection.recv.return_value = {"ssh_address_or_host": "jump-host.example"}
        fake_listener = mock.Mock()
        fake_listener.accept.return_value = fake_connection

        with (
            mock.patch(
                "multiprocessing.connection.Listener", return_value=fake_listener
            ),
            mock.patch.object(ssh_tunnel_worker, "_ssh_forward_process") as run_process,
            mock.patch.object(
                ssh_tunnel_worker, "_remove_unix_socket"
            ) as remove_socket,
        ):
            code = ssh_tunnel_worker.main("/tmp/seml_test_worker.sock", "aa")

        self.assertEqual(code, 0)
        run_process.assert_called_once_with(
            fake_connection, {"ssh_address_or_host": "jump-host.example"}
        )
        fake_connection.close.assert_called_once_with()
        fake_listener.close.assert_called_once_with()
        remove_socket.assert_called_once_with("/tmp/seml_test_worker.sock")

    def test_worker_main_rejects_invalid_authkey(self):
        code = ssh_tunnel_worker.main("/tmp/seml_test_worker.sock", "not-hex")
        self.assertEqual(code, 1)


class TestSSHForwardDispatcher(unittest.TestCase):
    def test_get_ssh_forward_parses_worker_output_and_caches(self):
        fake_proc = FakeWorkerProcess()
        fake_connection = FakeConnection(
            recv_values=[("127.0.0.1", 27018)], poll_values=[True]
        )
        ssh_config = {
            "ssh_address_or_host": "jump-host.example",
            "retries_max": 1,
            "retries_delay": 0.01,
        }

        with (
            mock.patch.object(ssh_forward, "assert_package_installed"),
            mock.patch.object(
                ssh_forward,
                "_start_ssh_forward_subprocess",
                return_value=(fake_proc, fake_connection, "/tmp/seml_test.sock"),
            ) as start_proc,
            mock.patch.object(ssh_forward.atexit, "register") as register_exit,
            mock.patch.dict(ssh_forward._forwards, {}, clear=True),
            mock.patch.dict(ssh_forward._forward_processes, {}, clear=True),
        ):
            first = ssh_forward._get_ssh_forward(ssh_config)
            second = ssh_forward._get_ssh_forward(ssh_config)

        self.assertEqual(first, ("127.0.0.1", 27018))
        self.assertEqual(second, first)
        start_proc.assert_called_once_with(0.04)
        register_exit.assert_called_once()
        self.assertEqual(fake_connection.sent, [ssh_config])

    def test_close_subprocess_sends_stop_and_terminates(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        connection = mock.Mock()

        with mock.patch.object(ssh_forward, "_remove_unix_socket") as remove_socket:
            ssh_forward._close_ssh_forward_subprocess(
                proc, connection, "/tmp/seml_test.sock"
            )

        connection.send.assert_called_once_with("stop")
        connection.close.assert_called_once_with()
        proc.terminate.assert_called_once_with()
        remove_socket.assert_called_once_with("/tmp/seml_test.sock")

    def test_start_subprocess_invokes_worker_module_and_connects(self):
        fake_proc = FakeWorkerProcess()
        fake_connection = object()

        with (
            mock.patch.object(
                ssh_forward.subprocess, "Popen", return_value=fake_proc
            ) as popen,
            mock.patch.object(
                ssh_forward, "_connect_to_ssh_worker", return_value=fake_connection
            ) as connect,
        ):
            proc, connection, socket_address = (
                ssh_forward._start_ssh_forward_subprocess(1.5)
            )

        self.assertIs(proc, fake_proc)
        self.assertIs(connection, fake_connection)
        self.assertEqual(socket_address, popen.call_args.args[0][4])

        command = popen.call_args.args[0]
        self.assertEqual(
            command[:4],
            [
                ssh_forward.sys.executable,
                "-m",
                "seml.utils.ssh_tunnel_worker",
                "--worker",
            ],
        )
        self.assertTrue(command[4].startswith("/tmp/seml_ssh_forward_"))
        self.assertTrue(command[4].endswith(".sock"))
        self.assertEqual(len(command[5]), 32)

        connect.assert_called_once_with(command[4], bytes.fromhex(command[5]), 1.5)
