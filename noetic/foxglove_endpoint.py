"""Check the local ROS1 bridge handshake and print the WLAN connection address."""
import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import time

PORT = 8766
PROTOCOL = "foxglove.websocket.v1"
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def handshake(host, port):
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Protocol: {PROTOCOL}\r\n\r\n"
    )
    with socket.create_connection((host, port), timeout=1.5) as sock:
        sock.sendall(request.encode("ascii"))
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk or len(data) + len(chunk) > 16384:
                raise ValueError("incomplete WebSocket response")
            data.extend(chunk)
        lines = bytes(data).split(b"\r\n\r\n", 1)[0].decode("latin1").split("\r\n")
        expected = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        headers = {name.lower(): value.strip() for name, value in
                   (line.split(":", 1) for line in lines[1:] if ":" in line)}
        if (lines[0].split()[1:2] != ["101"] or headers.get("sec-websocket-accept") != expected
                or headers.get("upgrade", "").lower() != "websocket"
                or "upgrade" not in {v.strip().lower() for v in headers.get("connection", "").split(",")}
                or headers.get("sec-websocket-protocol") != PROTOCOL):
            raise ValueError("port did not negotiate the Foxglove WebSocket protocol")
        mask = os.urandom(4)
        code = struct.pack("!H", 1000)
        sock.sendall(b"\x88\x82" + mask + bytes(a ^ b for a, b in zip(code, mask)))


def wlan_ips():
    result = subprocess.run(["ip", "-j", "-4", "addr", "show", "dev", "wlan0"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if result.returncode:
        return []
    return [ip["local"] for iface in json.loads(result.stdout)
            for ip in iface.get("addr_info", []) if ip.get("scope") == "global"]


def main():
    addresses = wlan_ips()
    host = next(iter(addresses), "127.0.0.1")
    deadline = time.monotonic() + 18
    while True:
        try:
            handshake(host, PORT)
            break
        except (OSError, ValueError) as exc:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Foxglove 在 {host}:{PORT} 握手失败：{exc}")
            time.sleep(.3)
    if addresses:
        print(f"Foxglove 本机握手成功。在电脑的 Foxglove WebSocket 连接中填 ws://{host}:{PORT}")
    else:
        print(f"Foxglove 本机握手成功。wlan0 未分配 IPv4，请在客户端使用可达的机器狗地址及端口 {PORT}。")


if __name__ == "__main__":
    main()
