#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from realtime_decoder import binary_record


def _find_rec_file(save_dir, prefix, manager_label, postfix):
    pattern = os.path.join(save_dir, f"{prefix}.*.{manager_label}.{postfix}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No rec files found for pattern: {pattern}"
        )
    if len(matches) > 1:
        print("Found multiple files, using the first one:")
        print(f"  {matches[0]}")
    return matches[0]


def _parse_rank_and_digits(filepath, manager_label, postfix):
    name = os.path.basename(filepath)
    parts = name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unexpected rec filename format: {name}")
    if parts[-2] != manager_label or parts[-1] != postfix:
        raise ValueError(f"Unexpected rec filename format: {name}")
    rank_str = parts[-3]
    num_digits = len(rank_str)
    return int(rank_str), num_digits


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
    prefix,
    save_dir=os.path.join("realtime_decoder", "output"),
    manager_label="state",
    postfix="bin_rec",
    sampling_rate=20000.0,
    lower=0.0,
    upper=360.0,
    show=True,
    csv_path=None,
):
    rec_path = _find_rec_file(save_dir, prefix, manager_label, postfix)
    rank, num_digits = _parse_rank_and_digits(rec_path, manager_label, postfix)

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
    if csv_path:
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
    parser.add_argument("prefix", help="Record file prefix (e.g. 20260207_225807_offline)")
    parser.add_argument(
        "--save-dir",
        default=os.path.join("realtime_decoder", "output"),
        help="Directory containing the .rec files (default: realtime_decoder/output)"
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
        help="Optional path to save decoded variables as CSV (default: None)"
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
    )


if __name__ == "__main__":
    main()
