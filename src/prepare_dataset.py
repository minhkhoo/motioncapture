"""
prepare_dataset.py — Generate / update metadata.csv from the "Zipped data" dataset
==================================================================================
MERGE mode, NOT overwrite: re-running when new sessions arrive PRESERVES every
existing row (especially the good_recording and notes you filled in by hand) and
only appends new rows.

The "Zipped data" tree supports BOTH layouts:
  - Old: Zipped data/Primary sessions/Amatures/<name>_<date>_<time>_primary/
  - New (from promote_session.py): Zipped data/Amatures/<name>_<date>_<time>/
A session = any folder containing camera_data.csv, at any depth.
"""

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT.parent / "Zipped data"
OUTPUT_DIR = PROJECT_ROOT / "prepared_dataset"

# Only a {date}_{time} stamp is required in the name — the primary/secondary suffix
# is NOT required (that classification is assigned after reviewing the video, not
# known at recording time).
SESSION_RE = re.compile(r"^(?P<player>.+)_(?P<date>\d{8})_(?P<time>\d{6})(?:_(?:primary|secondary))?$")
VIDEO_CANDIDATES = ("annotated_video.mp4", "raw_video.mp4")

# NEW schema — dropped dominant_hand / device_wrist / camera_view (the project only
# records male right-handed athletes). Legacy columns still in metadata.csv are KEPT.
FIELDNAMES = [
    "attempt_id", "player_id", "skill_level", "session_type", "session_date",
    "imu_path", "camera_path", "video_path", "good_recording", "notes",
]


def read_session_info(session_dir: Path) -> dict:
    p = session_dir / "session_info.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [WARN] session_info.json is corrupt, skipping: {session_dir.name}")
    return {}


def derive_skill(session_dir: Path, info: dict) -> str:
    """skill_level by priority: session_info.json → parent folder → '' (blank)."""
    kind = info.get("session_kind")
    if kind in ("professional", "amateur"):
        return kind
    parts = [p.lower() for p in session_dir.relative_to(SOURCE_DIR).parts]
    if any("proffesional" in p for p in parts):
        return "professional"
    if any("amature" in p for p in parts):
        return "amateur"
    return ""   # can't infer → leave blank, will be listed for manual entry


def derive_session_type(session_dir: Path) -> str:
    """primary/secondary if the path (or name) says so; otherwise 'unrated'."""
    parts = [p.lower() for p in session_dir.relative_to(SOURCE_DIR).parts]
    if any("primary" in p for p in parts):
        return "primary"
    if any("secondary" in p for p in parts):
        return "secondary"
    return "unrated"


def find_sessions(source_dir: Path):
    n_test = 0
    for session_dir in sorted(source_dir.rglob("*")):
        if not session_dir.is_dir() or not (session_dir / "camera_data.csv").exists():
            continue
        info = read_session_info(session_dir)
        # (b) Skip test sessions
        if info.get("session_kind") == "test" or session_dir.name.startswith("TEST_"):
            n_test += 1
            continue
        match = SESSION_RE.match(session_dir.name)
        if not match:
            print(f"  [skip] folder name doesn't match {{name}}_{{date}}_{{time}}: {session_dir.name}")
            continue
        yield session_dir, match, info
    if n_test:
        print(f"  [test] skipped {n_test} test session(s)")


def compute_identity(session_dir: Path, match, info: dict) -> tuple[str, str, str, str, str]:
    """Cheap, no file copy. Returns (attempt_id, player, skill, stype, session_date)."""
    player = info.get("player_id") or match.group("player")
    date_str = match.group("date")
    timestamp = f"{date_str}_{match.group('time')}"
    skill = derive_skill(session_dir, info)
    stype = derive_session_type(session_dir)
    attempt_id = f"{skill or 'unknown'}_{stype}_{player}_{timestamp}"
    session_date = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
    return attempt_id, player, skill, stype, session_date


def build_row(session_dir: Path, attempt_id, player, skill, stype, session_date) -> dict:
    """Copy the 3 files with the attempt_id prefix, return the default row for a NEW session."""
    camera_dst = OUTPUT_DIR / f"{attempt_id}_camera.csv"
    imu_dst = OUTPUT_DIR / f"{attempt_id}_imu.csv"
    shutil.copy2(session_dir / "camera_data.csv", camera_dst)
    imu_name = ""
    if (session_dir / "imu_data.csv").exists():
        shutil.copy2(session_dir / "imu_data.csv", imu_dst)
        imu_name = imu_dst.name

    video_src = next((session_dir / n for n in VIDEO_CANDIDATES
                      if (session_dir / n).exists()), None)
    video_name = ""
    if video_src is not None:
        video_dst = OUTPUT_DIR / f"{attempt_id}_video.mp4"
        shutil.copy2(video_src, video_dst)
        video_name = video_dst.name

    return {
        "attempt_id": attempt_id,
        "player_id": player,
        "skill_level": skill,
        "session_type": stype,          # (d) defaults to 'unrated' when the path is silent
        "session_date": session_date,
        "imu_path": imu_name,
        "camera_path": camera_dst.name,
        "video_path": video_name,
        "good_recording": "yes",
        "notes": "",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUTPUT_DIR / "metadata.csv"

    # (a) Read the old metadata, indexed by attempt_id. KEEP every legacy column
    # (including dropped ones like dominant_hand — old data is not lost).
    existing: dict[str, dict] = {}
    old_fieldnames: list[str] = []
    if meta_path.exists():
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            for r in reader:
                existing[r["attempt_id"]] = r

    kept, added, need_skill = 0, 0, []
    for session_dir, match, info in find_sessions(SOURCE_DIR):
        aid, player, skill, stype, sdate = compute_identity(session_dir, match, info)
        if aid in existing:
            kept += 1                 # ALREADY PRESENT → keep entirely, don't re-copy
            continue
        existing[aid] = build_row(session_dir, aid, player, skill, stype, sdate)
        added += 1
        if skill == "":
            need_skill.append(aid)

    # Column order: new schema + any leftover legacy columns (so old-column data isn't lost)
    fieldnames = list(FIELDNAMES)
    for c in old_fieldnames:
        if c not in fieldnames:
            fieldnames.append(c)

    # (a) Back up before writing
    if meta_path.exists():
        shutil.copy2(meta_path, meta_path.with_suffix(".csv.bak"))

    rows = sorted(existing.values(), key=lambda r: r["attempt_id"])
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in fieldnames})

    print(f"\nMERGE: kept {kept} existing row(s), added {added} new "
          f"→ {len(rows)} rows total in {meta_path.name}")
    if meta_path.with_suffix('.csv.bak').exists():
        print(f"  (backed up the old version: {meta_path.with_suffix('.csv.bak').name})")
    if need_skill:
        print(f"\n  ⚠️  {len(need_skill)} NEW session(s) had no inferable skill_level "
              "— please fill in the skill_level column by hand:")
        for aid in need_skill:
            print(f"       - {aid}")


if __name__ == "__main__":
    main()
