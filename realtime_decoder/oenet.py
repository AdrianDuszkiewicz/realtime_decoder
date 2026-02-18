# oenet.py
import zmq
import struct
import time
import numpy as np
import pandas as pd
import struct
from realtime_decoder import messages
from realtime_decoder.base import DataSourceReceiver
from realtime_decoder.datatypes import (
    Datatypes,
    LFPPoint,
    SpikePoint,
    CameraModulePoint,
)

# -------------------
# Head-direction CSV
# -------------------
hd_samples = None
hd_values = None
hd_index = None
_hd_warned = False


def load_hd_csv(csv_path):
    global hd_samples, hd_values, hd_index
    df = pd.read_csv(csv_path, usecols=[0, 1]).ffill().bfill()
    hd_samples = df.iloc[:, 0].astype(int).to_numpy()
    hd_values = (np.degrees(df.iloc[:, 1].to_numpy()) + 360) % 360
    hd_index = None

# -------------------
# Spike packet decoding
# -------------------
N_CHANNELS = 4
N_SAMPLES = 40
waveform_buf = np.empty((N_CHANNELS, N_SAMPLES), dtype=np.float32)
threshold_buf = np.empty((N_CHANNELS,), dtype=np.float32)

spike_header_fmt = "<BBHHHqdh"
spike_header_size = struct.calcsize(spike_header_fmt)
spike_expected_size = (
    spike_header_size + (N_CHANNELS * 4) + (N_CHANNELS * N_SAMPLES * 4)
)

lfp_header_fmt = "<BBHHHqdII"
lfp_header_size = struct.calcsize(lfp_header_fmt)

event_base_fmt = "<BBHHHqd"
event_base_size = struct.calcsize(event_base_fmt)
ttl_fmt = "<BBQ"
ttl_size = struct.calcsize(ttl_fmt)

def parse_spike(data, sample_rate=20000):
    if len(data) < spike_expected_size:
        return None
    (
        event_type, electrode_type, source_proc, source_stream,
        source_elec, sample_num, timestamp, sorted_id
    ) = struct.unpack_from(spike_header_fmt, data, 0)

    offset = spike_header_size
    np.copyto(
        threshold_buf,
        np.frombuffer(data, dtype=np.float32, count=N_CHANNELS, offset=offset),
    )
    offset += 4 * N_CHANNELS
    np.copyto(
        waveform_buf.ravel(),
        np.frombuffer(
            data, dtype=np.float32, count=N_CHANNELS * N_SAMPLES, offset=offset
        ),
    )

    return {
        "source_elec": source_elec,
        "sample_num": sample_num,
        "waveforms": waveform_buf.copy(),
    }


def parse_lfp(buf):
    """
    Minimal Falcon LFP parser.
    Extracts sample_num, n_samples, sample_rate.
    Ignores the float samples for low latency.
    """

    # Root table offset
    table_start = struct.unpack_from("<I", buf, 0)[0]

    # Helper: get field offset from vtable
    def field_offset(field_num):
        vtable_start = table_start - struct.unpack_from("<H", buf, table_start)[0]
        vtable_size = struct.unpack_from("<H", buf, vtable_start)[0]
        if field_num * 2 + 4 >= vtable_size:
            return 0
        return struct.unpack_from("<H", buf, vtable_start + 4 + field_num * 2)[0]

    # field numbers from schema:
    # 3 = n_channels, 4 = n_samples, 5 = sample_num, 6 = timestamp,
    # 7 = message_id, 8 = sample_rate

    def read_field(fmt, field_num, default=0):
        off = field_offset(field_num)
        return struct.unpack_from(fmt, buf, table_start + off)[0] if off else default

    n_samples   = read_field("<I", 4)
    sample_num  = read_field("<Q", 5)
    sample_rate = read_field("<I", 8)

    return {
        "sample_num": sample_num,
        "n_samples": n_samples,
        "sample_rate": sample_rate,
    }


def parse_ttl(data, sample_rate=20000):
    if len(data) < event_base_size + ttl_size:
        return None
    (
        base_type, sub_type, proc_id, stream_id, chan_idx,
        sample_num, timestamp
    ) = struct.unpack_from(event_base_fmt, data, 0)
    line, state, word = struct.unpack_from(ttl_fmt, data, event_base_size)
    return {
        "kind": "ttl",
        "processor_id": proc_id,
        "stream_id": stream_id,
        "sample_num": sample_num,
        "timestamp": sample_num / sample_rate,
        "line": line,
        "state": state,
        "word": word,
    }


# -------------------
# OEClient
# -------------------
class OEClient:
    # MAKE SURE THE ADDRESS MATCHES THAT IN OPENEPHYS FALCON
    def __init__(
            self,
            event_broadcaster_port=5557,
            falcon_output_port=5555,
            sample_rate=20000,
            csv_path=None,
    ):

        self.sample_rate = sample_rate
        if csv_path:
            load_hd_csv(csv_path)

        ctx = zmq.Context.instance()
        event_broadcaster_addr = f"tcp://127.0.0.1:{event_broadcaster_port}"
        falcon_output_addr = f"tcp://127.0.0.1:{falcon_output_port}"

        # spikes/TTL
        self.socket = ctx.socket(zmq.SUB)
        self.socket.connect(event_broadcaster_addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVHWM, 10)

        # LFP
        self.lfp_socket = ctx.socket(zmq.SUB)
        self.lfp_socket.connect(falcon_output_addr)
        self.lfp_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.lfp_socket.setsockopt(zmq.RCVHWM, 10)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.lfp_socket, zmq.POLLIN)

    def next_event(self):
        global hd_index
        try:
            socks = dict(self.poller.poll(1))
        except zmq.ZMQError:
            return None

        if self.socket in socks:  # spikes + TTL
            topic, data = self.socket.recv_multipart()
            if topic == b"\x01\x00":
                return parse_spike(data, sample_rate=self.sample_rate)
            elif topic == b"\x00\x00":
                ev = parse_ttl(data, sample_rate=self.sample_rate)
                #print(f"[DEBUG OEClient] Got TTL line={ev['line']} state={ev['state']} sample_num={ev['sample_num']}")
                if ev is None or ev["state"] == 0:
                    return None
                if ev["line"] == 4:
                    global _hd_warned
                    if hd_samples is None:
                        if not _hd_warned:
                            #print("[OEClient] Head-direction CSV not loaded; set openephys.csv_path in config.")
                            _hd_warned = True
                        return None
                    if hd_index is None:
                        hd_index = np.searchsorted(hd_samples, ev["sample_num"])
                        #print(f"[OEClient] Head direction index init at {hd_index}")
                    hd_time_sec = ev["sample_num"] / float(self.sample_rate)
                    hd_idx = np.searchsorted(hd_samples, ev["sample_num"])
                    hd_val = hd_values[hd_idx] if hd_idx < len(hd_values) else float("nan")
                    #print(
                    #    f"[DEBUG OEClient] HD TTL time={hd_time_sec:.3f}s hd={hd_val:.2f}"
                    #)
                    return ("hd", ev["sample_num"])
                elif ev["line"] == 3:
                    return ("lfp", ev["sample_num"])
            return None

        if self.lfp_socket in socks:  # LFP continuous data
            data = self.lfp_socket.recv()
            ev = parse_lfp(data)
            lfp_rate = ev.get("sample_rate") or self.sample_rate
            lfp_time_sec = ev["sample_num"] / float(lfp_rate)
            #print(
                #f"[DEBUG OEClient] LFP time={lfp_time_sec:.3f}s "
               # f"sample_num={ev['sample_num']} n_samples={ev['n_samples']}"
            #)
            return ev

        return None

    def receive(self):
        ev = self.next_event()
        if ev is None:
            time.sleep(0.01)
            return
        if isinstance(ev, (messages.StartupSignal, messages.TerminateSignal)):
            return ev
        return



# -------------------
# OEDataReceiver
# -------------------
class OEDataReceiver(DataSourceReceiver):
    def __init__(self, comm, rank, config, datatype):
        if datatype not in (Datatypes.LFP, Datatypes.SPIKES, Datatypes.LINEAR_POSITION):
            raise TypeError(f"Invalid datatype {datatype}")
        super().__init__(comm, rank, config, datatype)

        openephys_config = config["openephys"]
        self.client = OEClient(
            event_broadcaster_port=openephys_config["event_broadcaster_port"],
            falcon_output_port=openephys_config["falcon_output_port"],
            sample_rate=config["sampling_rate"]["spikes"],
            csv_path=openephys_config.get("csv_path"),
        )

        self.start = True

    def __next__(self):
        if not self.start:
            return None

        if self.datatype == Datatypes.SPIKES:
            ev = self.client.next_event()   # from Event Broadcaster
            if isinstance(ev, dict) and "source_elec" in ev:
                #print(f"[OEDataReceiver] Waveform shape: {ev["waveforms"].shape}")
                #print(f"[OEDataReceiver] Got spike from source_elec={ev['source_elec']}, sample_num={ev['sample_num']}")
                return SpikePoint(
                    ev["sample_num"],
                    ev["source_elec"],
                    ev["waveforms"] *-1, # ENCODER NEEDS POSITIVE WAVEFORMS
                    0,
                    time.time_ns(),
                )
            return None

        elif self.datatype == Datatypes.LFP:
            # only check Falcon socket
            if self.client.lfp_socket in dict(self.client.poller.poll(1)):
                data = self.client.lfp_socket.recv()
                ev = parse_lfp(data)
                #print(f"[DEBUG OEDataReceiver] Sending LFPPoint sample_num={ev['sample_num']} n_samples={ev['n_samples']}")
                return LFPPoint(
                    ev["sample_num"],
                    [0],
                    [0.0],
                    0,
                    time.time_ns(),
                )
            return None

        elif self.datatype == Datatypes.LINEAR_POSITION:
            ev = self.client.next_event()   # from Event Broadcaster TTLs
            if isinstance(ev, tuple) and ev[0] == "hd":
                if hd_samples is None:
                    return None
                sample_num = ev[1]
                idx = np.searchsorted(hd_samples, sample_num, side="left")
                hd_val = hd_values[idx] if idx < len(hd_values) else 0.0
                #print(f"[DEBUG OEClient] Got HD packet sample_num={sample_num} HD={hd_val}")
                return CameraModulePoint(
                    timestamp=sample_num,
                    segment=0,
                    position=int(hd_val),
                    x=0,
                    y=0,
                    x2=0,
                    y2=0,
                    t_recv_data=time.time_ns(),
                )
            return None

        return None

    def register_datatype_channel(self, datatype, channel=None):
        return

    def activate(self):
        self.start = True

    def deactivate(self):
        self.start = False

    def stop_iterator(self):
        raise StopIteration()
