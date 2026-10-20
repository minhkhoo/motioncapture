#!/usr/bin/env python3
"""
build_tracking.py — unified dataset tracking sheet (single source of truth)
=======================================================================
Merges two kinds of columns into ONE file, `data_tracking.xlsx`:

  • [AUTO]   — quality re-scored by code on every run: imu_verdict
               (NO when the IMU file is missing / dead baseline / loss > threshold), freq,
               loss, stalls, clip, accel/gyro peaks, fps, gap, detect, sync, overlap,
               actual_issues, skill_level + session_type (from the Zipped path /
               session_info.json).
  • [MANUAL] — filled in by hand; the script NEVER overwrites them: contact_ts,
               landed_ts, good_recording, quality_star, notes.

Scope: "Zipped data" (primary/secondary sessions) + named-athlete sessions under
recording/ (professional/amateur, skipping error-tagged ones) — new sessions are picked
up automatically.

Run:
    uv run --with openpyxl python build_tracking.py
After running: updates `data_tracking.xlsx` (manual columns preserved) and writes
`manual_contacts.csv` (read by extraction.py).

imu_verdict is "yes" or "no (reason)"; sync_cam_first is "yes" or "no".
"""
import os, re, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent
RECORDING  = ROOT / "recording"
# "Zipped data" now lives INSIDE motioncapture/ (moved from the parent folder on
# 2026-08-26). Prefer the new location; fall back to the old parent-level spot only
# if the new one doesn't exist, so this keeps working if it's ever moved back.
ZIPPED     = (ROOT / "Zipped data") if (ROOT / "Zipped data").exists() else ROOT.parent / "Zipped data"
TRACK_XLSX = ROOT / "data_tracking.xlsx"
MANUAL_CSV = ROOT / "manual_contacts.csv"

LOSS_DEAD = 50    # loss above this -> treated as NO usable IMU (IMU-related features unreliable)
RATE_MIN_HZ = 30  # effective sampling rate below this -> not usable
DEAD_ACC_G = 2.0      # peak acceleration below this ...
DEAD_GYRO_DPS = 100   # ... or peak angular velocity below this -> the sensor did not move (dead baseline)

MANUAL_COLS = ["contact_ts", "landed_ts", "good_recording", "quality_star", "notes"]
SKILL_FROM_FOLDER = {"amatures": "amateur", "proffesionals": "professional"}

# Fallback when Zipped data has been flattened (Primary/Secondary × Amatures/Proffesionals
# folders removed, sessions sit directly under "Zipped data/") — SKILL_FROM_FOLDER then has
# nothing to match. Derived from the ORIGINAL folder hierarchy (verified no player straddled
# both skill groups): Hieu & Hoang were under Proffesionals, Thinh & Vang under Amatures.
SKILL_FROM_PLAYER = {"hieu": "professional", "hoang": "professional",
                     "thinh": "amateur", "vang": "amateur"}


# ──────────────────────────────────────────────────────────────────────
# QUALITY SCORING (drops NaN rows, detects dead IMU / high loss)
# ──────────────────────────────────────────────────────────────────────
def quality_metrics(folder: Path) -> dict:
    camp = folder / "camera_data.csv"
    imup = folder / "imu_data.csv"
    if not imup.exists():
        # a few sessions keep a renamed IMU file (e.g. imu_data_deadbaseline.csv)
        alt = sorted(folder.glob("imu_data*.csv"))
        if alt:
            imup = alt[0]
    r = {}
    # ---- CAMERA ----
    camok = False
    try:
        dc = pd.read_csv(camp)
        ts = dc["timestamp"].dropna().values
        if len(dc) == 0 or len(ts) < 2:
            r.update(cam_fps=0, cam_median_gap_ms=np.nan, cam_max_gap_ms=np.nan,
                     detect_pct=np.nan, detect_frames="0/0", pass1=0, pass2=0, pass0=0)
        else:
            camok = True
            cg = np.diff(ts)
            ndet = len(dc) - (dc["wrist_x"].isna().sum() if "wrist_x" in dc else 0)
            vc = dc["detection_pass"].value_counts().to_dict() if "detection_pass" in dc else {}
            r.update(cam_fps=round((len(ts) - 1) / (ts[-1] - ts[0]), 1),
                     cam_median_gap_ms=round(float(np.median(cg)) * 1000, 1),
                     cam_max_gap_ms=round(float(cg.max()) * 1000),
                     detect_pct=round(100 * ndet / len(dc)),
                     detect_frames=f"{ndet}/{len(dc)}",
                     pass1=int(vc.get(1, 0)), pass2=int(vc.get(2, 0)), pass0=int(vc.get(0, 0)))
    except Exception:
        r.update(cam_fps=0, cam_median_gap_ms=np.nan, cam_max_gap_ms=np.nan,
                 detect_pct=np.nan, detect_frames="error", pass1=0, pass2=0, pass0=0)
    # video frame count (catches empty videos)
    r["video_frames"] = np.nan
    try:
        import cv2
        v = folder / "raw_video.mp4"
        if v.exists():
            c = cv2.VideoCapture(str(v)); r["video_frames"] = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    except Exception:
        pass
    # ---- IMU ----
    hasfile = imup.exists() and imup.stat().st_size > 50
    if not hasfile:
        for k in ("imu_freq_hz", "imu_stalls", "imu_max_stall_ms", "imu_loss_pct", "lost_stall",
                  "lost_small", "accel_range_G", "accel_clip_pct", "accel_peak_G",
                  "gyro_clip_pct", "gyro_peak_dps"):
            r[k] = np.nan
        r["imu_verdict"] = "no (missing file)"
    else:
        di = pd.read_csv(imup)
        at = di["arduino_time"].values.astype(np.int64) / 1e6; at = at - at[0]
        dur = at.max() - at.min(); dat = np.diff(at)
        freq = len(di) / dur if dur > 0 else 0
        stalls = dat[dat > 0.1]; maxstall = dat.max() * 1000 if len(dat) else 0
        exp = int(dur / 0.01); loss = 100 * (exp - len(di)) / exp if exp > 0 else 0
        small = dat[(dat > 0.020) & (dat <= 0.1)]
        lost_stall = int(np.sum(np.round(stalls / 0.01) - 1)); lost_small = int(np.sum(np.round(small / 0.01) - 1))
        acc = np.sqrt(di["ax"]**2 + di["ay"]**2 + di["az"]**2); gy = np.sqrt(di["gx"]**2 + di["gy"]**2 + di["gz"]**2)
        accpk = float(acc.max()); gypk = float(gy.max())
        ax = max(np.abs(di["ax"]).max(), np.abs(di["ay"]).max(), np.abs(di["az"]).max())
        rng = 24 if ax > 8.5 else (8 if ax > 4.05 else 4)
        ceil = 23.99 if rng == 24 else (7.99 if rng == 8 else 3.99)
        clip = 100 * ((np.abs(di["ax"]) >= ceil) | (np.abs(di["ay"]) >= ceil) | (np.abs(di["az"]) >= ceil)).sum() / len(di)
        # Per-axis (Part C2 fix): checking the combined magnitude >=2000 overcounts
        # — three axes at ~1200 dps each already give a magnitude over 2000 with no
        # single axis anywhere near its real +-2000 dps ceiling. Gyro FS never
        # changes across firmware versions (Part C1 audit: no .ino calls setGyroFS).
        gclip = 100 * ((np.abs(di["gx"]) >= 1999.0) | (np.abs(di["gy"]) >= 1999.0) | (np.abs(di["gz"]) >= 1999.0)).sum() / len(di)
        # Checks in order: rate, loss, dead baseline (the missing-file check is above).
        if round(freq, 1) < RATE_MIN_HZ:
            verdict = f"no (rate {freq:.0f} Hz<{RATE_MIN_HZ})"
        elif loss > LOSS_DEAD:
            verdict = f"no (loss {loss:.0f}%>{LOSS_DEAD})"
        elif accpk < DEAD_ACC_G or gypk < DEAD_GYRO_DPS:
            verdict = "no (dead baseline)"
        else:
            verdict = "yes"
        r.update(imu_freq_hz=round(freq, 1), imu_stalls=int(len(stalls)), imu_max_stall_ms=round(maxstall),
                 imu_loss_pct=round(loss), lost_stall=lost_stall, lost_small=lost_small,
                 accel_range_G=rng, accel_clip_pct=round(clip, 1), accel_peak_G=round(accpk, 1),
                 gyro_clip_pct=round(gclip, 1), gyro_peak_dps=round(gypk), imu_verdict=verdict)
    # ---- SYNC ----
    if hasfile and camok:
        di = pd.read_csv(imup); pc = di["pc_time"].dropna().values
        ts = pd.read_csv(camp)["timestamp"].dropna().values
        cf = bool(ts.min() < pc.min())
        r["sync_cam_first"] = "yes" if cf else "no"
        r["sync_delay_s"] = round(abs(float(ts.min() - pc.min())), 2)
        r["overlap_pct"] = round(100 * (min(ts.max(), pc.max()) - max(ts.min(), pc.min())) / (ts.max() - ts.min()))
    else:
        r.update(sync_cam_first="—", sync_delay_s=np.nan, overlap_pct=np.nan)
    # ---- REAL ISSUES (low detect is ignored: athlete left the frame, not a fault) ----
    p = []
    if not str(r["imu_verdict"]).startswith("yes"):
        p.append(r["imu_verdict"].replace("no ", "", 1))
    lv = r.get("imu_loss_pct")
    if lv == lv and lv is not None and 20 < lv <= LOSS_DEAD:
        p.append(f"loss {lv}%")
    ms = r.get("imu_max_stall_ms")
    if ms == ms and ms is not None and ms >= 500:
        p.append(f"stall {ms:.0f}ms")
    cl = r.get("accel_clip_pct")
    if cl == cl and cl is not None and cl > 5:
        p.append(f"clip {cl}%")
    ov = r.get("overlap_pct")
    if ov == ov and ov is not None and ov < 70:
        p.append(f"overlap {ov}%")
    if r.get("sync_cam_first") == "no":
        p.append("imu-before-camera")
    vf = r.get("video_frames")
    if vf == vf and vf is not None and vf < 5:
        p.append("empty video")
    r["actual_issues"] = "; ".join(p) if p else "OK"
    return r


# ──────────────────────────────────────────────────────────────────────
# COLLECT SESSIONS
# ──────────────────────────────────────────────────────────────────────
def collect_sessions() -> list[dict]:
    """Return a list of {session_name, folder, source, skill_level, session_type, player_id}."""
    sessions = []
    seen = set()   # keyed by player_ts to avoid Zipped <-> recording duplicates

    def key(name):  # player_ts, type suffix removed
        m = re.match(r"^(.+?_\d{8}_\d{6})", name)
        return m.group(1) if m else name

    # 1) Zipped data (has primary/secondary)
    if ZIPPED.exists():
        for camp in ZIPPED.rglob("camera_data.csv"):
            folder = camp.parent
            name = folder.name
            parts = [q.lower() for q in folder.relative_to(ZIPPED).parts]
            session_type = "primary" if any("primary" in q for q in parts) else \
                           ("secondary" if any("secondary" in q for q in parts) else "")
            skill = next((SKILL_FROM_FOLDER[q] for q in parts if q in SKILL_FROM_FOLDER), "")
            if not skill:   # flat layout: no Amatures/Proffesionals folder level to read
                skill = SKILL_FROM_PLAYER.get(name.split("_")[0].lower(), "")
                if not skill:
                    print(f"  [WARN] Could not determine skill_level for '{name}' "
                          f"(unknown player, not in SKILL_FROM_PLAYER) — left blank.")
            sessions.append(dict(session_name=name, folder=folder, source="zipped",
                                 skill_level=skill, session_type=session_type,
                                 player_id=name.split("_")[0]))
            seen.add(key(name))

    # 2) recording/ named-athlete (professional/amateur), skipping error-tagged
    for folder in sorted(RECORDING.iterdir() if RECORDING.exists() else []):
        if not folder.is_dir():
            continue
        name = folder.name
        if "error" in name.lower():
            continue
        if not (folder / "camera_data.csv").exists():
            continue
        info_path = folder / "session_info.json"
        if not info_path.exists():
            continue
        try:
            kind = json.loads(info_path.read_text(encoding="utf-8")).get("session_kind")
        except Exception:
            kind = None
        if kind not in ("professional", "amateur"):
            continue          # skip test/anonymous
        if key(name) in seen:
            continue          # already in Zipped
        sessions.append(dict(session_name=name, folder=folder, source="recording",
                             skill_level=kind, session_type="new",   # primary/secondary not assigned yet
                             player_id=name.split("_")[0]))
        seen.add(key(name))
    return sessions


# ──────────────────────────────────────────────────────────────────────
# SEED / KEEP MANUAL COLUMNS
# ──────────────────────────────────────────────────────────────────────
def _norm_landed(name_cols, df):
    """Normalise the landed_ts column name (old files may say 'landed_ts (foot touches ground)')."""
    for c in df.columns:
        if str(c).strip().lower().startswith("landed_ts"):
            return c
    return None


def load_manual_layer(path: Path, is_excel: bool) -> dict:
    """Read the MANUAL columns from one file (tracking xlsx or manual_contacts.csv) -> {session: {col: val}}."""
    out = {}
    if not path.exists():
        return out
    try:
        df = pd.read_excel(path) if is_excel else pd.read_csv(path)
    except Exception:
        return out
    if "session_name" not in df.columns:
        return out
    landed_col = _norm_landed(df.columns, df)
    for _, row in df.iterrows():
        s = str(row["session_name"]).strip()
        if not s or s == "nan":
            continue
        d = {}
        # contact_ts
        if "contact_ts" in df.columns and pd.notna(row.get("contact_ts")):
            d["contact_ts"] = row["contact_ts"]
        # landed_ts (xlsx) or landing_ts (manual_contacts.csv)
        if landed_col and pd.notna(row.get(landed_col)):
            d["landed_ts"] = row[landed_col]
        elif "landing_ts" in df.columns and pd.notna(row.get("landing_ts")):
            d["landed_ts"] = row["landing_ts"]
        for c in ("good_recording", "quality_star", "notes"):
            if c in df.columns and pd.notna(row.get(c)):
                d[c] = row[c]
        if d:
            out[s] = d
    return out


def merge_manual(sessions) -> dict:
    """Merge manual columns by priority: project xlsx > manual_contacts.csv."""
    layers = [
        load_manual_layer(MANUAL_CSV, is_excel=False),   # lowest priority
        load_manual_layer(TRACK_XLSX, is_excel=True),    # highest priority (the maintained file)
    ]
    manual = {}
    for layer in layers:
        for s, d in layer.items():
            manual.setdefault(s, {}).update(d)
    return manual


# ──────────────────────────────────────────────────────────────────────
# EXPORT manual_contacts.csv (for extraction.py)
# ──────────────────────────────────────────────────────────────────────
def _parse_ts(v):
    """Normalise one hand-entered timestamp to (float_value, approximate):
        '4.56'          -> (4.56, 0)      single number, certain
        '~4.237'        -> (4.237, 1)     leading ~ means approximate
        '5.644-5.675'   -> (5.6595, 1)    RANGE -> take the midpoint, mark approximate
        empty/'nan'      -> (None, 0)
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None, 0
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None, 0
    approx = 1 if s.startswith("~") else 0
    s = s.lstrip("~").strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$", s)   # range a-b
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2.0, 1
    try:
        return float(s), approx
    except ValueError:
        return None, approx


def emit_manual_contacts(df: pd.DataFrame):
    rows, skipped = [], []
    for _, r in df.iterrows():
        raw = str(r.get("contact_ts", "")).strip()
        has_text = raw not in ("", "nan", "None")
        ct, c_approx = _parse_ts(r.get("contact_ts"))
        if ct is None:
            if has_text:                       # contact written but unreadable -> WARN
                skipped.append((r["session_name"], raw))
            continue
        lt, l_approx = _parse_ts(r.get("landed_ts"))
        rows.append(dict(session_name=r["session_name"], contact_ts=round(ct, 3),
                         landing_ts=("" if lt is None else round(lt, 3)),
                         landing_approximate=int(bool(c_approx or l_approx)),
                         source="manual_coleman",
                         notes=("" if pd.isna(r.get("notes")) else r.get("notes"))))
    out = pd.DataFrame(rows, columns=["session_name", "contact_ts", "landing_ts",
                                      "landing_approximate", "source", "notes"])
    out.to_csv(MANUAL_CSV, index=False)
    if skipped:
        print(f"\n⚠️  {len(skipped)} contact_ts unreadable (entered by hand but badly formatted) — "
              f"NOT exported; fix them in data_tracking.xlsx:")
        for s, v in skipped:
            print(f"     {s}: '{v}'")
    return len(out)


# ──────────────────────────────────────────────────────────────────────
def main():
    print("Collecting sessions (Zipped + recording named-athlete)...")
    sessions = collect_sessions()
    print(f"  → {len(sessions)} session "
          f"({sum(s['source']=='zipped' for s in sessions)} zipped, "
          f"{sum(s['source']=='recording' for s in sessions)} recording)")

    manual = merge_manual(sessions)

    print("Scoring quality...")
    rows = []
    for i, s in enumerate(sessions, 1):
        q = quality_metrics(s["folder"])
        row = dict(session_name=s["session_name"], source=s["source"], player_id=s["player_id"],
                   skill_level=s["skill_level"], session_type=s["session_type"])
        row.update({c: manual.get(s["session_name"], {}).get(c, "") for c in MANUAL_COLS})
        row.update(q)
        rows.append(row)
        if i % 20 == 0 or i == len(sessions):
            print(f"\r  {i}/{len(sessions)}", end="", flush=True)
    print()

    ident = ["session_name", "source", "player_id", "skill_level", "session_type"]
    auto = ["imu_verdict", "imu_freq_hz", "imu_stalls", "imu_max_stall_ms", "imu_loss_pct",
            "lost_stall", "lost_small", "accel_range_G", "accel_clip_pct", "accel_peak_G",
            "gyro_clip_pct", "gyro_peak_dps", "cam_fps", "cam_median_gap_ms", "cam_max_gap_ms",
            "video_frames", "detect_pct", "detect_frames", "pass1", "pass2", "pass0",
            "sync_cam_first", "sync_delay_s", "overlap_pct", "actual_issues"]
    cols = ident + MANUAL_COLS + auto
    df = pd.DataFrame(rows)[cols].sort_values(["source", "player_id", "session_name"])

    with pd.ExcelWriter(TRACK_XLSX, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="tracking")
        ws = w.sheets["tracking"]
        ws.freeze_panes = "F2"   # freeze identifier columns + header
        for i, c in enumerate(cols, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = max(11, min(26, len(c) + 2))
    n_emit = emit_manual_contacts(df)

    marked = int((df["contact_ts"].astype(str).str.strip().replace("nan", "") != "").sum())
    print(f"\n✅ Wrote {TRACK_XLSX.name}: {len(df)} sessions x {len(cols)} columns")
    print(f"   Manual columns preserved: {', '.join(MANUAL_COLS)}")
    print(f"   Rows with contact_ts : {marked}/{len(df)}  -> exported {n_emit} rows to {MANUAL_CSV.name}")
    print(f"   imu_verdict       : {df['imu_verdict'].apply(lambda s: str(s).split(' (')[0]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
