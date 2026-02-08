import zmq
import time

#How to interpret output:
#If you see multipart: topic=b'\x01\x00' ... → you’re on the spike port.
#If you see single: bytes=... → you’re on Falcon (or any single-frame PUB).
#If nothing in 5 seconds → nothing is publishing (or acquisition isn’t running).

HOST = "127.0.0.1"
PORT = 3333  # change me
addr = f"tcp://{HOST}:{PORT}"

ctx = zmq.Context.instance()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.SUBSCRIBE, b"")
s.setsockopt(zmq.RCVTIMEO, 5000)
s.connect(addr)

print(f"Listening on {addr}...")

try:
    while True:
        # Try multipart first (spike broadcaster style)
        try:
            parts = s.recv_multipart(flags=zmq.NOBLOCK)
            if len(parts) == 2:
                topic, data = parts
                print(f"multipart: topic={topic!r} data_bytes={len(data)}")
            else:
                sizes = [len(p) for p in parts]
                print(f"multipart: {len(parts)} parts sizes={sizes}")
        except zmq.Again:
            # Fall back to single-frame (Falcon style)
            msg = s.recv()
            print(f"single: bytes={len(msg)}")
except zmq.Again:
    print("No messages in 5 seconds.")
