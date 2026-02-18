#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from realtime_decoder import binary_record


def _resolve_save_dir(save_dir):
    """Resolve common relative save_dir variants to an existing directory."""
    if save_dir is None:
        save_dir = "output"

    expanded = os.path.expanduser(save_dir)
    candidates = []

    def _add(path):
        if path and path not in candidates:
            candidates.append(path)

    _add(expanded)
    _add(os.path.abspath(expanded))

    normalized = expanded.replace("\\", "/")
    legacy_prefix = "realtime_decoder/output/"
    if normalized == "realtime_decoder/output":
        _add("output")
        _add(os.path.abspath("output"))
    elif normalized.startswith(legacy_prefix):
        _add(normalized[len("realtime_decoder/"):])
        _add(os.path.abspath(normalized[len("realtime_decoder/"):]))

    for path in candidates:
        if os.path.isdir(path):
            return path

    return expanded


def _find_rec_file(save_dir, prefix, manager_label, postfix, rank=None):
    if prefix:
        if rank is None:
            pattern = os.path.join(save_dir, f"{prefix}.*.{manager_label}.{postfix}")
        else:
            pattern = os.path.join(save_dir, f"{prefix}.{rank}.{manager_label}.{postfix}")
    else:
        if rank is None:
            pattern = os.path.join(save_dir, f"*.{manager_label}.{postfix}")
        else:
            pattern = os.path.join(save_dir, f"*.{rank}.{manager_label}.{postfix}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        available = []
        if os.path.isdir(save_dir):
            available = sorted(
                f for f in os.listdir(save_dir)
                if f.endswith(f".{manager_label}.{postfix}")
            )
        details = (
            f" Available {manager_label}.{postfix} files: {available[:5]}"
            if available else ""
        )
        raise FileNotFoundError(
            f"No rec files found for pattern: {pattern}.{details}"
        )
    if prefix is None and len(matches) > 1:
        raise ValueError(
            f"Multiple rec files found in {save_dir}; specify prefix."
        )
    if prefix and len(matches) > 1:
        print("Found multiple files, using the first one:")
        print(f"  {matches[0]}")
    return matches[0]


def _parse_prefix_rank_and_digits(filepath, manager_label, postfix):
    name = os.path.basename(filepath)
    parts = name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unexpected rec filename format: {name}")
    if parts[-2] != manager_label or parts[-1] != postfix:
        raise ValueError(f"Unexpected rec filename format: {name}")
    rank_str = parts[-3]
    num_digits = len(rank_str)
    prefix = ".".join(parts[:-3])
    return prefix, int(rank_str), num_digits


def _extract_pos_columns(columns):
    pos_cols = []
    bin_ids = []
    state_labels = set()
    for col in columns:
        if not col.startswith("x"):
            continue
        # format: x<bin>_<state>
        try:
            bin_part, state_part = col[1:].split("_", 1)
            bin_id = int(bin_part)
        except ValueError:
            continue
        pos_cols.append(col)
        bin_ids.append(bin_id)
        state_labels.add(state_part)
    if not pos_cols:
        raise ValueError("No posterior columns found (expected x<bin>_<state>).")
    num_bins = max(bin_ids) + 1
    num_states = len(state_labels)
    return pos_cols, num_bins, num_states


def plot_decoder_hd(
    prefix=None,
    save_dir="output",
    manager_label="state",
    postfix="bin_rec",
    sampling_rate=20000.0,
    lower=0.0,
    upper=360.0,
    show=True,
    csv_path=None,
    rank=None,
):
    save_dir = _resolve_save_dir(save_dir)
    rec_path = _find_rec_file(save_dir, prefix, manager_label, postfix, rank=rank)
    if prefix is None:
        prefix, rank, num_digits = _parse_prefix_rank_and_digits(
            rec_path, manager_label, postfix
        )
    else:
        _, rank, num_digits = _parse_prefix_rank_and_digits(
            rec_path, manager_label, postfix
        )

    reader = binary_record.BinaryRecordsFileReader(
        save_dir, prefix, rank, num_digits,
        manager_label, postfix
    )
    frames = reader.convert_to_pandas()
    dec_id = int(binary_record.RecordIDs.DECODER_OUTPUT)
    if dec_id not in frames or frames[dec_id].empty:
        raise ValueError("DECODER_OUTPUT record not found or empty.")

    df = frames[dec_id]
    if "mapped_pos" not in df.columns:
        raise ValueError("Expected 'mapped_pos' column in DECODER_OUTPUT records.")

    pos_cols, num_bins, num_states = _extract_pos_columns(df.columns)
    posterior = df[pos_cols].to_numpy()
    decoded_flat = np.argmax(posterior, axis=1)
    decoded_bin = decoded_flat % num_bins

    bin_edges = np.linspace(lower, upper, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    decoded_angle = bin_centers[decoded_bin]

    real_angle = np.asarray(df["mapped_pos"]) % 360.0
    diff = ((decoded_angle - real_angle + 180.0) % 360.0) - 180.0

    time_sec = np.asarray(df["timestamp"]) / sampling_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time_sec, decoded_angle, label="Decoded HD", linewidth=1)
    axes[0].plot(time_sec, real_angle, label="Real HD", linewidth=1)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_ylim(0, 360)
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[0].legend(loc="upper right")

    axes[1].plot(time_sec, diff, label="HD Diff", linewidth=1)
    axes[1].set_ylabel("Diff (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-180, 180)
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    if csv_path is None:
        csv_path = os.path.join(save_dir, f"{prefix}_hd.csv")
    csv_df = df.copy()
    csv_df["decoded_angle"] = decoded_angle
    csv_df["real_angle"] = real_angle
    csv_df["hd_diff"] = diff
    csv_df["time_sec"] = time_sec
    csv_df.to_csv(csv_path, index=False)
    if show:
        plt.show()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot decoded HD, real HD, and signed difference from a .rec file."
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default=None,
        help="Record file prefix (omit to auto-detect if only one rec file exists)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Optional decoder rank to select a specific rec file",
    )
    parser.add_argument(
        "--save-dir",
        default="output",
        help="Directory containing the .rec files (default: output)"
    )
    parser.add_argument(
        "--manager-label",
        default="state",
        help="Binary record manager label (default: state)"
    )
    parser.add_argument(
        "--postfix",
        default="bin_rec",
        help="Binary record postfix (default: bin_rec)"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=20000.0,
        help="Timestamp sampling rate in Hz (default: 20000)"
    )
    parser.add_argument(
        "--lower",
        type=float,
        default=0.0,
        help="Lower bound for decoded angle bins (default: 0)"
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=360.0,
        help="Upper bound for decoded angle bins (default: 360)"
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional path to save decoded variables as CSV (default: save_dir/<prefix>_hd.csv)"
    )
    args = parser.parse_args()

    plot_decoder_hd(
        args.prefix,
        save_dir=args.save_dir,
        manager_label=args.manager_label,
        postfix=args.postfix,
        sampling_rate=args.sampling_rate,
        lower=args.lower,
        upper=args.upper,
        show=True,
        csv_path=args.csv_path,
        rank=args.rank,
    )


if __name__ == "__main__":
    main()
