"""
build_features_tracking.py — build features_tracking.csv from the non-error sessions
==========================================================================================
Filters data_tracking.xlsx to sessions where contact_ts is NOT 'error'. landed_ts is NOT
used by any current feature (see extraction.py: "Reserved for a future follow-through
measurement; not used by the model yet"), so a session with only landed_ts='error' is
kept — only a bad contact_ts excludes a session.

Runs the existing extraction.py pipeline (manual_contacts.csv contacts) on each session
folder, and writes one row per session with exactly these columns:

  session_name, player, skill_level, session_date, camera_fps, detect_pct,
  cam_angle_at_contact_deg, cam_angle_max_cocking_deg, cam_angle_rom_deg,
  cam_angle_rate_acceleration_dps, cam_wrist_path_length_norm,
  cam_wrist_displacement_norm, cam_wrist_peak_speed_ratio_per_sec,
  phase_acceleration_duration_s

Replaces features_tracking.csv in place.
"""
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from extraction import process_session, find_sessions_recursive

OUT_COLS = [
    "session_name", "player", "skill_level", "session_date", "camera_fps", "detect_pct",
    "cam_angle_at_contact_deg", "cam_angle_max_cocking_deg", "cam_angle_rom_deg",
    "cam_angle_rate_acceleration_dps", "cam_wrist_path_length_norm",
    "cam_wrist_displacement_norm", "cam_wrist_peak_speed_ratio_per_sec",
    "phase_acceleration_duration_s",
]

DATE_RE = re.compile(r"_(\d{8})_\d{6}")


def is_error(x) -> bool:
    return str(x).strip().lower() == "error"


def session_date_from_name(name: str) -> str:
    m = DATE_RE.search(name)
    if not m:
        return ""
    d = m.group(1)  # YYYYMMDD
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def main():
    xlsx_path = ROOT / "data_tracking.xlsx"
    df_meta = pd.read_excel(xlsx_path)

    c_err = df_meta["contact_ts"].apply(is_error)
    keep_mask = ~c_err
    target = df_meta[keep_mask].copy()
    print(f"Sessions with contact_ts != 'error' (landed_ts not used by any feature, ignored): {len(target)}")

    # A blank (NaN) contact_ts is not the literal string 'error' but still cannot be
    # extracted (no known contact time) — exclude it here and report it explicitly.
    blank_contact = target["contact_ts"].isna()
    if blank_contact.any():
        for _, row in target[blank_contact].iterrows():
            print(f"  [SKIP] {row['session_name']}: contact_ts is blank (not marked yet) — "
                  f"cannot extract, excluded from output")
        target = target[~blank_contact]

    print(f"Sessions to extract: {len(target)}")

    # Locate every session folder once (recording/ + "Zipped data/", recursive)
    sessions, _ = find_sessions_recursive(ROOT)
    folder_by_name = {p.name: p for p in sessions}

    rows = []
    meta_by_name = target.set_index("session_name")

    for name, meta_row in meta_by_name.iterrows():
        folder = folder_by_name.get(name)
        if folder is None:
            print(f"  [SKIP] {name}: session folder not found on disk")
            continue

        feats_list = process_session(folder, verbose=False)
        if not feats_list:
            print(f"  [SKIP] {name}: extraction produced no contact/features")
            continue
        if len(feats_list) > 1:
            print(f"  [WARN] {name}: {len(feats_list)} contacts detected, using the first")
        feats = feats_list[0]

        rows.append({
            "session_name": name,
            "player": meta_row.get("player_id"),
            "skill_level": meta_row.get("skill_level"),
            "session_date": session_date_from_name(name),
            "camera_fps": meta_row.get("cam_fps"),
            "detect_pct": meta_row.get("detect_pct"),
            "cam_angle_at_contact_deg": feats.get("cam_angle_at_contact_deg"),
            "cam_angle_max_cocking_deg": feats.get("cam_angle_max_cocking_deg"),
            "cam_angle_rom_deg": feats.get("cam_angle_rom_deg"),
            "cam_angle_rate_acceleration_dps": feats.get("cam_angle_rate_acceleration_dps"),
            "cam_wrist_path_length_norm": feats.get("cam_wrist_path_length_norm"),
            "cam_wrist_displacement_norm": feats.get("cam_wrist_displacement_norm"),
            "cam_wrist_peak_speed_ratio_per_sec": feats.get("cam_wrist_peak_speed_ratio_per_sec"),
            "phase_acceleration_duration_s": feats.get("phase_acceleration_duration_s"),
        })

    out_df = pd.DataFrame(rows, columns=OUT_COLS)
    out_path = ROOT / "features_tracking.csv"
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved {len(out_df)} rows x {len(OUT_COLS)} columns -> {out_path}")


if __name__ == "__main__":
    main()
