"""
promote_session.py — Move ONE session from recording/ into the "Zipped data" dataset
====================================================================================
Automates the mechanical, easy-to-forget steps: copy the session into the correct
dataset branch and update metadata.csv. Does NOT automate the judgment steps
(reviewing the video, rating quality, assigning primary/secondary) — those are still
done by hand afterwards.

Usage:
  python src/promote_session.py recording/Khoa_20260720_120000
  python src/promote_session.py recording/OldSession_... --skill professional
  python src/promote_session.py recording/TestNew_... --force        # force a test session
  python src/promote_session.py recording/... --dry-run              # preview, change nothing

After promoting, continue with:
  python extraction.py --root "../Zipped data" --output features.csv --quiet
  python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIPPED = PROJECT_ROOT.parent / "Zipped data"
# Keep the existing (misspelled) folder names so prepare_dataset's skill detection matches.
SKILL_FOLDER = {"professional": "Proffesionals", "amateur": "Amatures"}


def die(msg: str):
    sys.exit(f"[REFUSED] {msg}")


def main():
    ap = argparse.ArgumentParser(description="Promote one session into the Zipped data dataset")
    ap.add_argument("session", help="Path to the session folder inside recording/")
    ap.add_argument("--skill", choices=["professional", "amateur"], default=None,
                    help="Required if the session has no session_info.json")
    ap.add_argument("--force", action="store_true",
                    help="Force-promote even a test session (refused by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print what would happen; no copy, no metadata change")
    args = ap.parse_args()

    src = Path(args.session)
    if not src.is_dir():
        die(f"not a directory: {src}")
    if not (src / "camera_data.csv").exists():
        die(f"missing camera_data.csv — not a valid session: {src}")

    # ── Read session_info.json ───────────────────────────────────────
    info = {}
    info_path = src / "session_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            print(f"[WARN] session_info.json is corrupt, treating it as absent.")

    kind = info.get("session_kind")

    # ── Test session → refuse unless --force ─────────────────────────
    is_test = (kind == "test") or src.name.startswith("TEST_")
    if is_test and not args.force:
        die("this is a test session, not added to the dataset. Use --force if you really mean it.")
    if is_test and args.force:
        print("[FORCE] promoting a test session on request (--force).")

    # ── Determine skill ──────────────────────────────────────────────
    skill = None
    if kind in ("professional", "amateur"):
        skill = kind
    if args.skill:
        if skill and args.skill != skill:
            print(f"[WARN] --skill '{args.skill}' DIFFERS from session_info.json '{skill}'. "
                  f"Using --skill as you requested.")
        skill = args.skill
    if skill is None:
        die("unknown skill_level (session has no session_info.json). "
            "Pass --skill professional|amateur.")

    dst = ZIPPED / SKILL_FOLDER[skill] / src.name

    print(f"  Source : {src}")
    print(f"  Dest   : {dst}")
    print(f"  Skill  : {skill}" + ("  (from --skill)" if args.skill else "  (from session_info.json)"))

    if dst.exists():
        die(f"destination already exists, will NOT overwrite: {dst}\n"
            "  To replace it, delete the destination folder manually first.")

    if args.dry_run:
        print("\n[DRY-RUN] would copy the whole folder to the destination, then run prepare_dataset.py.")
        print("          Nothing was changed.")
        return

    # ── Copy the whole folder (including session_info.json) ──────────
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"\n✅ Copied to: {dst}")

    # ── Update metadata via prepare_dataset.py (merge mode) ──────────
    print("\n→ Updating metadata.csv ...")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "prepare_dataset.py")])
    if r.returncode != 0:
        sys.exit("[ERROR] prepare_dataset.py failed — please check.")
    print("\nDone. Remember to re-run extraction.py + train_model.py to include the new session.")


if __name__ == "__main__":
    main()
