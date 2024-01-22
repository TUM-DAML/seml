import array
import fcntl
import socket
import struct
import sys
from contextlib import closing


def get_network_interfaces():
    # From https://code.activestate.com/recipes/439093/#c1
    is_64bits = sys.maxsize > 2**32
    struct_size = 40 if is_64bits else 32
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    max_possible = 8  # initial value
    while True:
        _bytes = max_possible * struct_size
        names = array.array('B')
        for i in range(0, _bytes):
            names.append(0)
        outbytes = struct.unpack(
            'iL',
            fcntl.ioctl(
                s.fileno(),
                0x8912,  # SIOCGIFCONF
                struct.pack('iL', _bytes, names.buffer_info()[0]),
            ),
        )[0]
        if outbytes == _bytes:
            max_possible *= 2
        else:
            break
    namestr = names.tobytes()
    ifaces = {}
    for i in range(0, outbytes, struct_size):
        iface_name = bytes.decode(namestr[i : i + 16]).split('\0', 1)[0]
        iface_addr = socket.inet_ntoa(namestr[i + 20 : i + 24])
        ifaces[iface_name] = iface_addr

    return ifaces


def find_free_port():
    ifaces = get_network_interfaces()
    if 'lo' in ifaces:
        del ifaces['lo']
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((list(ifaces.values())[0], 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ip_address, port = s.getsockname()[:2]
        return ip_address, port
