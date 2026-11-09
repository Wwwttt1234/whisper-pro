"""Apply the no-download preference inside each owned Python worker."""
import os
import socket
import ipaddress


def enable():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    original = socket.getaddrinfo

    def local_lookup(host, *args, **kwargs):
        if host not in {None, "localhost"}:
            try:
                allowed = ipaddress.ip_address(host).is_loopback
            except ValueError:
                allowed = False
            if not allowed:
                raise OSError("الاتصال الخارجي ممنوع في وضع دون تنزيل")
        return original(host, *args, **kwargs)

    socket.getaddrinfo = local_lookup
