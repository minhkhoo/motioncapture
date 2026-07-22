"""
train_model.py — Volleyball spike skill classification (professional vs amateur)
============================================================================
Step 10 (player-wise data split) · 11 (model ladder) · 12 (evaluation)

GOAL: build the CORRECT pipeline so it can be re-run when more data arrives — NOT
to obtain a pretty accuracy. The current dataset has only 4 players, far too few to
draw any conclusion.

Usage:
  python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv
  python src/train_model.py features.csv --metadata metadata.csv --exclude-low-quality
  python src/train_model.py features.csv --metadata metadata.csv --session-type primary
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MIN_SAMPLES = 20        # stop below this threshold
MIN_PLAYERS_TRUST = 6   # loud warning below this threshold


# ══════════════════════════════════════════════════════════════════════
# COLUMNS EXCLUDED FROM X — with reasons
# ══════════════════════════════════════════════════════════════════════

EXCLUDE_IDENTIFIER = {
    "session_name", "player_id", "skill_level", "session_type", "attempt_id",
    "join_key",
}
EXCLUDE_TECHNICAL = {
    "has_imu", "imu_timebase", "contact_frame", "cam_frame_number",
}
# Quality flags: KEEP in the dataframe for filtering, do NOT feed into X
EXCLUDE_QUALITY = {
    "clip_fraction", "peak_is_clipped", "contact_near_gap",
    "sync_quality_ms", "sync_lowlat_fraction", "n_contacts_rejected",
}
# cam_vis_* = DATA QUALITY indicators, not technique features.
# Measured: pro 0.864 vs amateur 0.929 — skewed OPPOSITE to skill.
# Feeding them in makes the model learn recording conditions, not spike technique.
EXCLUDE_PREFIX = ("cam_vis_",)

# ── FEATURE SETS (to trace the source of the camera-angle confound) ───────
# Prefix filter, applied AFTER all exclusions in select_features().
#   None = no extra filtering (keeps the default behavior).
FEATURE_SETS: dict[str, tuple[str, ...] | None] = {
    "all":         None,
    "imu_pure":    ("imu_",),
    "imu_phase":   ("imu_", "phase_"),
    "camera_only": ("cam_",),      # cam_vis_* already excluded earlier
}

FEATURE_SET_NOTES = {
    "imu_pure":
        "Sensor worn on the body → completely independent of camera placement.",
    "imu_phase":
        "⚠️  phase_* is NOT fully immune to the camera angle.\n"
        "     The cocking_end boundary is set by the ELBOW-ANGLE MAXIMUM,\n"
        "     and the elbow angle is a camera-derived quantity. So\n"
        "     phase_cocking_duration_s and phase_acceleration_duration_s DEPEND\n"
        "     INDIRECTLY on the camera angle. Only phase_followthrough_duration_s is\n"
        "     nearly independent. This is NOT a fully clean set.",
    "camera_only":
        "2D projection → depends directly on camera position/angle.",
}

SESSION_RE = re.compile(r"^(?P<player>.+?)_(?P<ts>\d{8}_\d{6})(?:_(?P<type>primary|secondary))?$")
# The type token accepts ANY label (primary/secondary/unrated/unknown) — new sessions
# from promote_session.py have session_type='unrated', so don't hardcode the old 2 values.
ATTEMPT_RE = re.compile(
    r"^(?P<skill>professional|amateur)_(?P<type>[a-z]+)_"
    r"(?P<player>.+?)_(?P<ts>\d{8}_\d{6})$")


def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


# ══════════════════════════════════════════════════════════════════════
# DATA SOURCE — metadata.csv is the SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════════

def _join_key(value: str) -> str | None:
    """Normalize to '{player}_{timestamp}'.

    features.csv uses session_name  = '{player}_{ts}_{type}'
    metadata.csv uses attempt_id    = '{skill}_{type}_{player}_{ts}'
    The two formats DIFFER, so a direct name join matches 0 rows.
    (player, timestamp) is unique, so it works as the key.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = ATTEMPT_RE.match(s)
    if m:
        return f"{m.group('player').lower()}_{m.group('ts')}"
    m = SESSION_RE.match(s)
    if m:
        return f"{m.group('player').lower()}_{m.group('ts')}"
    return s.lower()


def load_and_join(features_path: Path, metadata_path: Path) -> pd.DataFrame:
    feat = pd.read_csv(features_path)
    meta = pd.read_csv(metadata_path)

    hr("DATA SOURCE (metadata.csv is the single source of truth)")
    print(f"  features.csv : {len(feat)} rows, {feat.shape[1]} columns")
    print(f"  metadata.csv : {len(meta)} rows")

    required = {"player_id", "skill_level", "session_type", "good_recording"}
    missing = required - set(meta.columns)
    if missing:
        sys.exit(f"\n[ERROR] metadata.csv is missing required columns: {sorted(missing)}")
    if not ({"session_name", "attempt_id"} & set(meta.columns)):
        sys.exit("\n[ERROR] metadata.csv must have 'session_name' or 'attempt_id'")
    if "session_name" not in feat.columns:
        sys.exit("\n[ERROR] features.csv is missing the 'session_name' column")

    meta_key_col = "session_name" if "session_name" in meta.columns else "attempt_id"
    print(f"  join key     : features['session_name'] ↔ metadata['{meta_key_col}']")
    print(f"                 (normalized to '{{player}}_{{timestamp}}' since the two formats differ)")

    feat = feat.copy()
    meta = meta.copy()
    feat["join_key"] = feat["session_name"].map(_join_key)
    meta["join_key"] = meta[meta_key_col].map(_join_key)

    dup = meta["join_key"].duplicated().sum()
    if dup:
        print(f"  [WARN] metadata.csv has {dup} duplicate key(s) — keeping the first row")
        meta = meta.drop_duplicates("join_key", keep="first")

    meta_cols = ["join_key", "player_id", "skill_level", "session_type", "good_recording"]
    if "session_date" in meta.columns:          # needed for the per-session quality table (4c)
        meta_cols.append("session_date")
    merged = feat.merge(meta[meta_cols], on="join_key", how="left", validate="m:1")

    in_meta = merged["player_id"].notna()
    not_in_meta = merged.loc[~in_meta, "session_name"].unique()
    merged = merged[in_meta].copy()

    good = merged["good_recording"].astype(str).str.strip().str.lower() == "yes"
    dropped_bad = merged.loc[~good, "session_name"].unique()
    merged = merged[good].copy()

    print(f"\n  → matched metadata       : {len(merged) + sum(merged['session_name'].isin(dropped_bad))} "
          f"(dropped {len(not_in_meta)} session(s) not in metadata)")
    print(f"  → good_recording == yes  : {len(merged)} rows remaining "
          f"(dropped {len(dropped_bad)} session(s))")

    if len(not_in_meta):
        print(f"\n  Sessions DROPPED because not in metadata.csv ({len(not_in_meta)}):")
        for s in sorted(not_in_meta)[:40]:
            print(f"     - {s}")
        if len(not_in_meta) > 40:
            print(f"     ... and {len(not_in_meta) - 40} more session(s)")
    if len(dropped_bad):
        print(f"\n  Sessions DROPPED because good_recording != yes ({len(dropped_bad)}):")
        for s in sorted(dropped_bad):
            print(f"     - {s}")

    if merged.empty:
        sys.exit("\n[ERROR] No rows left after filtering by metadata.")

    print(f"\n  ✅ USED FOR TRAINING: {len(merged)} contacts "
          f"from {merged['session_name'].nunique()} session(s), "
          f"{merged['player_id'].nunique()} player(s)")
    return merged


# ══════════════════════════════════════════════════════════════════════
# STEP A — CAMERA-ANGLE CONFOUND CROSS-TABULATION
# ══════════════════════════════════════════════════════════════════════

def crosstab_report(df: pd.DataFrame) -> None:
    hr("STEP A — CAMERA-ANGLE CONFOUND: skill_level × session_type")
    print("  session_type (primary/secondary) is a CAMERA-ANGLE QUALITY RATING,")
    print("  not the target label — but it seeps into the camera features.\n")

    ct = pd.crosstab(df["skill_level"], df["session_type"])
    print("  CONTACT counts:")
    print(ct.to_string().replace("\n", "\n    ").rjust(0))

    print("\n  PLAYER count per cell:")
    pc = df.groupby(["skill_level", "session_type"])["player_id"].nunique().unstack(fill_value=0)
    print("    " + pc.to_string().replace("\n", "\n    "))

    print("\n  Player × camera angle (check whether they are nested):")
    pt = pd.crosstab(df["player_id"], df["session_type"])
    print("    " + pt.to_string().replace("\n", "\n    "))

    print("\n  Share of primary in each skill group:")
    pcts = {}
    for skill, g in df.groupby("skill_level"):
        pct = (g["session_type"] == "primary").mean() * 100
        pcts[skill] = pct
        print(f"    {skill:14s}: {pct:5.1f}% primary  (n={len(g)} contacts)")

    if len(pcts) == 2:
        gap = abs(list(pcts.values())[0] - list(pcts.values())[1])
        print(f"\n  Difference: {gap:.1f} percentage points")
        if gap > 20:
            print("  ⚠️  CONFOUND WARNING: difference > 20 percentage points.")
            print("     The camera-angle share is strongly skewed between the two skill")
            print("     groups → the model may learn the CAMERA ANGLE instead of TECHNIQUE.")
            print("     See the step B results below.")
        else:
            print("  → below the 20-point threshold. Still check the confound test (step B).")


# ══════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame) -> list[str]:
    hr("FEATURE PREPARATION")
    dropped = {"identifier / label": [], "technical": [], "quality flags": [],
               "cam_vis_* (recording quality)": [], "string type": [],
               "all NaN": [], "zero variance": []}
    cols = []

    for c in df.columns:
        if c in EXCLUDE_IDENTIFIER:
            dropped["identifier / label"].append(c); continue
        if c in EXCLUDE_TECHNICAL:
            dropped["technical"].append(c); continue
        if c in EXCLUDE_QUALITY:
            dropped["quality flags"].append(c); continue
        if any(c.startswith(p) for p in EXCLUDE_PREFIX):
            dropped["cam_vis_* (recording quality)"].append(c); continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            dropped["string type"].append(c); continue
        s = df[c]
        if s.isna().all():
            dropped["all NaN"].append(c); continue
        if s.nunique(dropna=True) <= 1:
            dropped["zero variance"].append(c); continue
        cols.append(c)

    print(f"  USED AS FEATURES ({len(cols)} columns):")
    for c in cols:
        na = df[c].isna().sum()
        print(f"     {c:42s} (NaN {na}/{len(df)})")

    print(f"\n  EXCLUDED FROM X:")
    for reason, names in dropped.items():
        if names:
            print(f"     [{reason}] ({len(names)}): {', '.join(sorted(names))}")

    if not cols:
        sys.exit("\n[ERROR] No features left to train on.")
    return cols


def apply_feature_set(all_cols: list[str], set_name: str,
                      verbose: bool = True) -> list[str]:
    """Filter the feature list by the prefixes of the chosen set.

    Applied AFTER select_features(), so all exclusions (cam_vis_*, quality flags,
    identifiers, constant columns) still hold.
    """
    prefixes = FEATURE_SETS[set_name]
    if prefixes is None:
        if verbose:
            print(f"  Feature set '{set_name}': using all {len(all_cols)} numeric columns.")
        return list(all_cols)

    kept = [c for c in all_cols if c.startswith(prefixes)]
    dropped = [c for c in all_cols if not c.startswith(prefixes)]

    if verbose:
        print(f"  Feature set '{set_name}': keeping {len(kept)}/{len(all_cols)} columns "
              f"(prefix {', '.join(prefixes)})")
        note = FEATURE_SET_NOTES.get(set_name)
        if note:
            print(f"     {note}")
        if dropped:
            print(f"     Excluded from this set ({len(dropped)}): {', '.join(dropped)}")
            # Columns with no prefix fall out of EVERY filter — say so clearly so the
            # user knows they only exist in 'all'.
            orphan = [c for c in dropped
                      if not c.startswith(("imu_", "cam_", "phase_"))]
            if orphan:
                print(f"     [NOTE] {len(orphan)} column(s) with no imu_/cam_/phase_ "
                      f"prefix belong to NO filter set, only to 'all': {', '.join(orphan)}")
                if "ipf_confidence" in orphan and set_name == "imu_pure":
                    print("             (ipf_confidence is actually an IMU quantity "
                          "— IPF = acc_dev × gyro_dev — but lacks the 'imu_' prefix, "
                          "so it is filtered out by the prefix rule.)")
    return kept


def quality_confound_check(df: pd.DataFrame) -> None:
    """Do the quality flags skew by session_type / skill_level?

    If they skew strongly, the confound comes from RECORDING CONDITIONS (different
    sessions, BLE distance, hardware) — not just from camera geometry.
    """
    hr("SIDE CHECK — DO THE QUALITY FLAGS SKEW BY GROUP?")
    cols = [c for c in ("clip_fraction", "sync_quality_ms", "sync_lowlat_fraction",
                        "cam_vis_wrist_mean") if c in df.columns]
    if not cols:
        print("  (no quality-flag columns in features.csv)")
        return

    for by in ("session_type", "skill_level"):
        if by not in df.columns or df[by].nunique() < 2:
            continue
        print(f"\n  Mean by {by}:")
        g = df.groupby(by)[cols].mean()
        print("    " + g.round(4).to_string().replace("\n", "\n    "))
        # Relative difference between the two groups
        if len(g) == 2:
            a, b = g.iloc[0], g.iloc[1]
            print(f"    → absolute difference ({g.index[0]} vs {g.index[1]}):")
            for c in cols:
                denom = max(abs(a[c]), abs(b[c]), 1e-9)
                rel = abs(a[c] - b[c]) / denom * 100
                mark = "  ← STRONG SKEW" if rel > 30 else ""
                print(f"        {c:24s} {abs(a[c]-b[c]):10.4f}  ({rel:5.1f}%){mark}")

    print("\n  Interpretation: if these flags skew strongly by session_type, then primary")
    print("  and secondary differ in RECORDING CONDITIONS (session, BLE distance,")
    print("  hardware) — the confound is not purely geometric, and filtering out camera")
    print("  features will NOT be enough to remove it.")


def quality_report(df: pd.DataFrame, do_filter: bool) -> pd.DataFrame:
    """Always print the quality table, whether filtering or not."""
    hr("QUALITY FILTER")
    n = len(df)
    flags = {}
    if "peak_is_clipped" in df:
        flags["peak_is_clipped == 1"] = (df["peak_is_clipped"] == 1)
    if "contact_near_gap" in df:
        flags["contact_near_gap == 1"] = (df["contact_near_gap"] == 1)
    if "sync_quality_ms" in df:
        flags["sync_quality_ms > 50"] = (df["sync_quality_ms"] > 50)

    print(f"  Total contacts: {n}")
    bad = pd.Series(False, index=df.index)
    for name, mask in flags.items():
        m = mask.fillna(False)
        bad |= m
        print(f"    {name:26s}: {int(m.sum()):4d} ({m.mean()*100:5.1f}%)")
    print(f"    {'HIT AT LEAST ONE FLAG':26s}: {int(bad.sum()):4d} ({bad.mean()*100:5.1f}%)")
    print(f"    {'remaining if filtered':26s}: {int((~bad).sum()):4d}")

    kept = df[~bad]
    print("\n  If filtered — remaining distribution by group / player:")
    if len(kept):
        print("    " + pd.crosstab(kept["player_id"], kept["skill_level"])
              .to_string().replace("\n", "\n    "))
    else:
        print("    (empty)")

    if not do_filter:
        print("\n  → --exclude-low-quality OFF: keeping everything, reporting only.")
        return df

    print("\n  → --exclude-low-quality ON: applying the filter.")
    return kept


def guard_dataset(df: pd.DataFrame, context: str) -> None:
    if len(df) < MIN_SAMPLES:
        sys.exit(f"\n[STOP] {context}: only {len(df)} samples left (< {MIN_SAMPLES}). "
                 "Not enough for meaningful training.")
    for label in ("skill_level",):
        if df[label].nunique() < 2:
            sys.exit(f"\n[STOP] {context}: only 1 {label} class left — cannot train.")
    per_class_players = df.groupby("skill_level")["player_id"].nunique()
    if (per_class_players < 2).any():
        sys.exit(
            f"\n[STOP] {context}: a class has only 1 player left:\n"
            f"{per_class_players.to_string()}\n"
            "  With LeaveOneGroupOut by player, dropping that player wipes out an\n"
            "  entire class from training → cannot train. Need more players for this class.")


# ══════════════════════════════════════════════════════════════════════
# STEP 11 — MODEL LADDER
# ══════════════════════════════════════════════════════════════════════

def build_models() -> dict:
    return {
        "Dummy": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
        ]),
        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "RandomForest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300,
                                           random_state=RANDOM_STATE)),
        ]),
    }


def run_task(df: pd.DataFrame, feat_cols: list[str], label_col: str,
             task_name: str, verbose: bool = True) -> tuple[list[dict], pd.DataFrame, dict]:
    """Run the 3-model ladder with LeaveOneGroupOut by player_id."""
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()
    groups = df["player_id"].to_numpy()

    logo = LeaveOneGroupOut()
    rows, importances = [], []
    pooled = {}

    if verbose:
        hr(f"STEP 10 — PLAYER-WISE SPLIT · task: predict {label_col}")
        n_players = df["player_id"].nunique()
        print(f"  LeaveOneGroupOut, groups = player_id → {n_players} folds")
        print("  A player NEVER appears in both train and test.\n")

    for name, model in build_models().items():
        y_true_all, y_pred_all = [], []
        for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
            test_player = np.unique(groups[te])[0]
            classes_te = np.unique(y[te])
            classes_tr = np.unique(y[tr])

            if len(classes_tr) < 2:
                if verbose:
                    print(f"    [SKIP FOLD] {name} fold {fold} (test={test_player}): "
                          f"train has only 1 class")
                continue

            m = model.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            y_true_all.append(y[te]); y_pred_all.append(pred)

            rows.append({
                "task": task_name, "model": name, "fold": fold,
                "test_player": test_player,
                "n_train": len(tr), "n_test": len(te),
                "n_classes_test": len(classes_te),
                "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                "f1_macro": f1_score(y[te], pred, average="macro", zero_division=0),
            })

            if name == "RandomForest":
                importances.append(m.named_steps["clf"].feature_importances_)

        if y_true_all:
            yt = np.concatenate(y_true_all); yp = np.concatenate(y_pred_all)
            pooled[name] = {
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(yt, yp).tolist(),
                "labels": sorted(np.unique(np.concatenate([yt, yp])).tolist()),
            }

    imp_df = pd.DataFrame()
    if importances:
        imp_df = (pd.DataFrame({"feature": feat_cols,
                                "importance": np.mean(importances, axis=0)})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    return rows, imp_df, pooled


def print_task_report(rows: list[dict], pooled: dict, task_name: str,
                      label_col: str) -> None:
    hr(f"STEP 12 — EVALUATION · task: predict {label_col}")
    res = pd.DataFrame(rows)
    res = res[res["task"] == task_name]
    if res.empty:
        print("  (no results)")
        return

    single = (res["n_classes_test"] == 1).all()
    if single:
        print("  ⚠️  EVERY TEST FOLD HAS ONLY 1 CLASS")
        print("     Each player belongs to exactly one class, so leaving that player")
        print("     out makes the test fold single-class. balanced_accuracy per fold then")
        print("     degenerates into that class's recall (a model always predicting one")
        print("     label scores 1.00 in one fold and 0.00 in the other). ⇒ READ THE POOLED")
        print("     numbers below; per-fold is only for seeing the variance.\n")

    print("  PER-FOLD RESULTS (balanced accuracy):")
    piv = res.pivot_table(index=["fold", "test_player", "n_test"],
                          columns="model", values="balanced_accuracy")
    print("    " + piv.round(3).to_string().replace("\n", "\n    "))

    print("\n  MEAN ± STD across folds:")
    for name, g in res.groupby("model"):
        b, f = g["balanced_accuracy"], g["f1_macro"]
        print(f"    {name:20s} bal_acc {b.mean():.3f} ± {b.std():.3f}   "
              f"f1_macro {f.mean():.3f} ± {f.std():.3f}")

    print("\n  POOLED (all fold predictions combined — THE KEY NUMBER):")
    for name, p in pooled.items():
        print(f"    {name:20s} bal_acc {p['balanced_accuracy']:.3f}   "
              f"f1_macro {p['f1_macro']:.3f}")

    print("\n  POOLED CONFUSION MATRIX (row = true, column = predicted):")
    for name, p in pooled.items():
        print(f"    {name} — labels {p['labels']}")
        for lab, row in zip(p["labels"], p["confusion_matrix"]):
            print(f"       {lab:14s} {row}")


def verdict(pooled: dict, res: pd.DataFrame, task_name: str) -> str:
    """Blunt conclusion: does it beat Dummy or not."""
    r = res[res["task"] == task_name]
    best = max((n for n in pooled if n != "Dummy"),
               key=lambda n: pooled[n]["balanced_accuracy"], default=None)
    if best is None or "Dummy" not in pooled:
        return "not enough data to conclude"

    bs = r[r["model"] == best]["balanced_accuracy"]
    ds = r[r["model"] == "Dummy"]["balanced_accuracy"]
    b_lo, b_hi = bs.mean() - bs.std(), bs.mean() + bs.std()
    d_lo, d_hi = ds.mean() - ds.std(), ds.mean() + ds.std()
    overlap = (b_lo <= d_hi) and (d_lo <= b_hi)

    print(f"\n  CONCLUSION:")
    print(f"    balanced accuracy: {best} {bs.mean():.2f} ± {bs.std():.2f} "
          f"vs Dummy {ds.mean():.2f} ± {ds.std():.2f}")
    if overlap:
        print("    → the spread OVERLAPS Dummy, NOT enough basis to conclude.")
        print("      A higher mean must NOT be read as 'the model works'.")
        return "overlaps Dummy - nothing proven yet"
    print("    → the spread is SEPARATE from Dummy (still needs more players to confirm).")
    return f"{best} separates from Dummy"


QUALITY_SIGNALS = ("sync_quality_ms", "imu_sample_rate", "clip_fraction")


def feature_quality_correlation(df: pd.DataFrame, feat_cols: list[str]) -> None:
    """(4a) Spearman of EACH feature against the recording-quality signals.

    Measures DIRECTLY instead of guessing: a feature that varies with recording
    quality (rather than technique) has a high |r|. |r| > 0.5 = QUALITY-CONTAMINATED.
    """
    hr("(4a) PER-FEATURE CORRELATION WITH RECORDING QUALITY (Spearman)")
    signals = [s for s in QUALITY_SIGNALS if s in df.columns]
    if not signals:
        print("  (no quality-signal columns)")
        return
    print(f"  Quality signals: {', '.join(signals)}")
    print("  |r| > 0.5 = the feature depends on recording conditions more than technique.\n")

    rows = []
    for c in feat_cols:
        rec = {"feature": c}
        best = 0.0
        for s in signals:
            if c == s:
                r = 1.0        # the feature IS the quality signal → perfect match
            else:
                sub = df[[c, s]].dropna()
                if len(sub) >= 3 and sub[c].nunique() > 1 and sub[s].nunique() > 1:
                    rr = spearmanr(sub[c], sub[s]).statistic
                    r = float(rr) if rr == rr else float("nan")
                else:
                    r = float("nan")
            rec[s] = r
            if r == r:
                best = max(best, abs(r))
        rec["max_abs_r"] = best
        rows.append(rec)

    tbl = pd.DataFrame(rows).sort_values("max_abs_r", ascending=False)
    hdr = f"  {'feature':40s}" + "".join(f"{s[:14]:>15s}" for s in signals) + f"{'max|r|':>9s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for _, r in tbl.iterrows():
        line = f"  {r['feature']:40s}" + "".join(f"{r[s]:>15.3f}" for s in signals)
        mark = "  ⚠️ CONTAMINATED" if r["max_abs_r"] > 0.5 else ""
        print(line + f"{r['max_abs_r']:>9.3f}" + mark)

    n_bad = int((tbl["max_abs_r"] > 0.5).sum())
    print(f"\n  → {n_bad}/{len(tbl)} feature(s) with |r| > 0.5 (quality-contaminated).")
    if n_bad:
        print("    Consider dropping these features, or read the model results cautiously.")


def quality_by_date(df: pd.DataFrame) -> None:
    """(4c) Quality table per session_date — for VIEWING only, not a label."""
    hr("(4c) QUALITY PER SESSION (session_date) — reference only")
    if "session_date" not in df.columns:
        print("  (metadata.csv has no session_date column)")
        return
    signals = [s for s in QUALITY_SIGNALS if s in df.columns]
    agg = {s: (s, "median") for s in signals}
    g = df.groupby("session_date").agg(n_contact=("session_name", "size"), **agg)
    print("  Median per session (n_contact = number of spikes):")
    print("    " + g.round(2).to_string().replace("\n", "\n    "))
    print("\n  This is NOT used as a label: quality by date depends on luck")
    print("  (signal interference, temperature), not a meaningful category.")


def _best_non_dummy(pooled: dict) -> float:
    return max((v["balanced_accuracy"] for k, v in pooled.items() if k != "Dummy"),
               default=float("nan"))


def compare_feature_sets(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    """Run all 4 feature sets, print a summary table + automatic interpretation."""
    hr("COMPARE 4 FEATURE SETS — TRACE THE CAMERA-ANGLE CONFOUND")
    print("  Hypothesis: the confound lives in cam_* (2D projection, camera-dependent),")
    print("  while imu_* is immune (sensor worn on the body).\n")

    summary, all_rows = [], []
    for set_name in FEATURE_SETS:
        print(f"\n  ── Set '{set_name}' " + "─" * 50)
        cols = apply_feature_set(all_cols, set_name, verbose=True)
        if len(cols) < 2:
            print(f"     [SKIP] only {len(cols)} column(s) left — not enough to train.")
            continue

        rows_s, _, pooled_s = run_task(df, cols, "skill_level",
                                       f"skill_level::{set_name}", verbose=False)
        skill = _best_non_dummy(pooled_s)
        dummy_s = pooled_s.get("Dummy", {}).get("balanced_accuracy", float("nan"))

        conf = float("nan")
        rows_c = []
        if df["session_type"].nunique() >= 2:
            rows_c, _, pooled_c = run_task(df, cols, "session_type",
                                           f"session_type::{set_name}", verbose=False)
            conf = _best_non_dummy(pooled_c)

        print(f"     → skill_level {skill:.3f} (Dummy {dummy_s:.3f})  |  "
              f"session_type {conf:.3f}")
        summary.append({"feature_set": set_name, "n_cols": len(cols),
                        "skill_level": skill, "session_type": conf,
                        "delta": skill - conf, "dummy_skill": dummy_s})
        all_rows.extend(rows_s + rows_c)

    if not summary:
        print("\n  No set could be run.")
        return pd.DataFrame()

    s = pd.DataFrame(summary)

    hr("SUMMARY TABLE (pooled balanced accuracy, best model excluding Dummy)")
    print(f"    {'feature set':<14}{'n_cols':>7}{'skill_level':>14}"
          f"{'session_type':>15}{'difference':>13}")
    print("    " + "─" * 63)
    for _, r in s.iterrows():
        print(f"    {r.feature_set:<14}{int(r.n_cols):>7}{r.skill_level:>14.3f}"
              f"{r.session_type:>15.3f}{r.delta:>13.3f}")
    print("\n    difference = skill_level − session_type.")
    print("    More NEGATIVE → the features lean toward encoding the CAMERA ANGLE over TECHNIQUE.")

    # ── AUTOMATIC INTERPRETATION ─────────────────────────────────────
    hr("INTERPRETATION")
    row = lambda n: s[s.feature_set == n].iloc[0] if (s.feature_set == n).any() else None
    imu, cam = row("imu_pure"), row("camera_only")

    if imu is not None and np.isfinite(imu.session_type):
        st = imu.session_type
        if 0.4 <= st <= 0.6:
            print(f"  ✅ HYPOTHESIS CONFIRMED: imu_pure predicts the camera angle {st:.3f} ≈ 0.5 "
                  "(chance level).")
            print("     The confound comes from cam_*. The imu_* set is CLEAN and usable.")
        elif st > 0.7:
            print(f"  ❌ HYPOTHESIS REJECTED: imu_pure STILL predicts the camera angle {st:.3f} "
                  "(>0.7).")
            print("     The confound is NOT only camera geometry. Another cause")
            print("     could be: primary and secondary were recorded on DIFFERENT SESSIONS,")
            print("     so they also differ in hardware, BLE distance, temperature.")
            print("     → CHECK: whether session_type correlates with clip_fraction /")
            print("       sync_quality_ms (the 'SIDE CHECK' table below). If it does,")
            print("       that is evidence for the recording-conditions hypothesis.")
        else:
            print(f"  ⚠️  imu_pure predicts the camera angle {st:.3f} — in between, no clear verdict.")
            print("     Part of the confound is beyond geometry; see the quality-flags table.")

    if cam is not None and np.isfinite(cam.session_type):
        if cam.session_type > 0.7:
            print(f"\n  ✅ camera_only predicts the camera angle {cam.session_type:.3f} (very high) →")
            print("     CONFIRMS cam_* is the main confound source.")
        else:
            print(f"\n  camera_only predicts the camera angle {cam.session_type:.3f} — not "
                  "unusually high.")

    best_skill = s.loc[s.skill_level.idxmax()] if s.skill_level.notna().any() else None
    if best_skill is not None:
        print(f"\n  Best set for skill_level: '{best_skill.feature_set}' "
              f"({best_skill.skill_level:.3f} vs Dummy {best_skill.dummy_skill:.3f})")
        if best_skill.skill_level <= best_skill.dummy_skill:
            print("     → still does NOT beat Dummy. No set proves anything about")
            print("       skill classification on the current dataset.")

    return pd.DataFrame(all_rows)


def check_importance_bias(imp: pd.DataFrame) -> None:
    if imp.empty:
        return
    hr("FEATURE IMPORTANCE (RandomForest, averaged across folds)")
    for i, r in imp.head(10).iterrows():
        print(f"    {i+1:2d}. {r['feature']:44s} {r['importance']:.4f}")

    def group_of(c):
        for p in ("cam_", "imu_", "phase_"):
            if c.startswith(p):
                return p
        return "other"

    top3 = imp.head(3)["feature"].tolist()
    groups = {group_of(c) for c in top3}
    if len(groups) == 1:
        g = groups.pop()
        print(f"\n  ⚠️  WARNING: all 3 most important features belong to group '{g}'")
        print(f"     ({', '.join(top3)})")
        if g == "cam_":
            print("     All camera features → the model may be latching onto RECORDING")
            print("     CONDITIONS (camera angle, distance, lighting) rather than spike technique.")
        else:
            print("     The model relies entirely on one signal source — check whether that")
            print("     source is systematically skewed between the two groups.")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Classify professional vs amateur from features.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", help="Path to features.csv")
    ap.add_argument("--metadata", required=True,
                    help="REQUIRED — metadata.csv, the single source of truth about "
                         "which sessions are used")
    ap.add_argument("--exclude-low-quality", action="store_true",
                    help="Drop contacts with peak_is_clipped=1 OR contact_near_gap=1 "
                         "OR sync_quality_ms>50")
    ap.add_argument("--session-type", choices=["primary", "secondary", "all"],
                    default="all", help="Train on one camera angle only (default all)")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="all",
                    help="Feature set to train on (default 'all' = original behavior). "
                         "imu_pure=imu_* only, imu_phase=imu_*+phase_*, "
                         "camera_only=cam_* only")
    ap.add_argument("--compare-feature-sets", action="store_true",
                    help="Run all 4 feature sets in sequence and print a summary table "
                         "to trace the camera-angle confound")
    ap.add_argument("--confound-label", choices=["session_type", "session_date", "none"],
                    default="session_type",
                    help="Label for the label-based confound test (default "
                         "session_type). 'none' to skip. A single-valued label is auto-skipped.")
    ap.add_argument("--confound-drop-unrated", action="store_true",
                    help="Drop sessions whose session_type is still 'unrated' instead of "
                         "stopping, so the confound test can run without rating them first.")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    fp, mp = Path(args.features), Path(args.metadata)
    if not fp.exists():
        sys.exit(f"[ERROR] features not found: {fp}")
    if not mp.exists():
        sys.exit(f"[ERROR] metadata not found: {mp}")

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    df = load_and_join(fp, mp)
    crosstab_report(df)

    # ── STEP C: filter by camera angle ───────────────────────────────
    if args.session_type != "all":
        hr(f"STEP C — CAMERA-ANGLE FILTER: keep '{args.session_type}' only")
        before = len(df)
        df = df[df["session_type"] == args.session_type].copy()
        print(f"  {before} → {len(df)} contacts")
        if len(df):
            print("    " + pd.crosstab(df["player_id"], df["skill_level"])
                  .to_string().replace("\n", "\n    "))
        guard_dataset(df, f"after filtering session_type={args.session_type}")

    df = quality_report(df, args.exclude_low_quality)
    guard_dataset(df, "after quality filtering")

    # ── Unrated-session guard for the confound test ──────────────────
    # The confound test predicts session_type; rows still marked 'unrated' cannot be
    # part of that. Require them rated first (or explicitly dropped). Only enforced
    # when session_type is actually used as a confound (compare mode, or the default
    # --confound-label session_type).
    uses_st_confound = args.compare_feature_sets or args.confound_label == "session_type"
    if uses_st_confound and "session_type" in df.columns:
        unrated_mask = df["session_type"].astype(str) == "unrated"
        if unrated_mask.any():
            unrated_sessions = sorted(df.loc[unrated_mask, "session_name"].unique())
            if args.confound_drop_unrated:
                df = df[~unrated_mask].copy()
                print(f"\n[--confound-drop-unrated] Dropped {len(unrated_sessions)} unrated "
                      f"session(s) ({int(unrated_mask.sum())} contacts) from the run.")
                guard_dataset(df, "after dropping unrated sessions")
            else:
                hr("STOP — UNRATED SESSIONS")
                print(f"  {len(unrated_sessions)} session(s) not yet rated primary/secondary. "
                      "Fill in the")
                print("  session_type column in metadata.csv before running the confound test,")
                print("  or use --confound-label none to skip this test.")
                print("  (Or use --confound-drop-unrated to drop them and run right away.)")
                print(f"\n  Unrated sessions ({len(unrated_sessions)}):")
                for s in unrated_sessions:
                    print(f"     - {s}")
                sys.exit(1)

    n_players = df["player_id"].nunique()
    if n_players < MIN_PLAYERS_TRUST:
        hr("⚠️  MAJOR WARNING — TOO FEW PLAYERS")
        print(f"  Only {n_players} players (trust threshold: {MIN_PLAYERS_TRUST}).")
        print("  With LeaveOneGroupOut, each fold drops 1 player = dropping 1/"
              f"{n_players} of the population.")
        print("  The results below are NOT RELIABLE ENOUGH TO DRAW CONCLUSIONS — this")
        print("  pipeline exists to be re-run when more players are added, not to")
        print("  publish accuracy.")
        print(f"  Distribution: {df.groupby('skill_level')['player_id'].nunique().to_dict()}"
              " players per class")

    all_feat_cols = select_features(df)

    # ── Compare-feature-sets mode (run, then exit) ───────────────────
    if args.compare_feature_sets:
        cmp_rows = compare_feature_sets(df, all_feat_cols)
        feature_quality_correlation(df, all_feat_cols)   # (4a)
        quality_confound_check(df)
        quality_by_date(df)                              # (4c)
        if not cmp_rows.empty:
            out = outdir / "results_feature_sets.csv"
            cmp_rows.to_csv(out, index=False)
            hr("OUTPUT")
            print(f"    ✅ {out}")
        return

    # ── Filter by the chosen feature set ─────────────────────────────
    hr(f"FEATURE SET: --feature-set {args.feature_set}")
    feat_cols = apply_feature_set(all_feat_cols, args.feature_set, verbose=True)
    if len(feat_cols) < 2:
        sys.exit(f"\n[STOP] Set '{args.feature_set}' has only {len(feat_cols)} column(s).")

    # (4a) Per-feature diagnostic via correlation with recording quality
    feature_quality_correlation(df, feat_cols)

    # ── Main task: skill_level ───────────────────────────────────────
    rows_skill, imp_skill, pooled_skill = run_task(
        df, feat_cols, "skill_level", "skill_level")
    print_task_report(rows_skill, pooled_skill, "skill_level", "skill_level")
    v_skill = verdict(pooled_skill, pd.DataFrame(rows_skill), "skill_level")
    check_importance_bias(imp_skill)

    # ── (4b) Label-based confound test (optional) ────────────────────
    conf_label = args.confound_label
    rows_conf, pooled_conf = [], {}
    hr(f"(4b) LABEL-BASED CONFOUND TEST: --confound-label {conf_label}")
    if conf_label == "none":
        print("  (skipped by request)")
    elif conf_label not in df.columns:
        print(f"  (skipped: metadata.csv has no '{conf_label}' column)")
    elif df[conf_label].nunique() < 2:
        vals = df[conf_label].dropna().unique()
        print(f"  (skipped: '{conf_label}' has only 1 value: {vals})")
    else:
        print(f"  Same features, label switched to '{conf_label}'. If this label is")
        print("  predicted well, the skill result above may just be the model latching")
        print("  onto recording conditions.\n")
        rows_conf, _, pooled_conf = run_task(df, feat_cols, conf_label, conf_label)
        print_task_report(rows_conf, pooled_conf, conf_label, conf_label)

    # ── Direct comparison of the two tasks ───────────────────────────
    hr("COMPARE THE TWO TASKS (pooled balanced accuracy)")
    best_of = lambda p: max((v["balanced_accuracy"] for k, v in p.items()
                             if k != "Dummy"), default=float("nan"))
    X_skill, Y_conf = best_of(pooled_skill), best_of(pooled_conf)
    print(f"    Predict skill_level        : balanced acc = {X_skill:.3f}")
    if pooled_conf:
        print(f"    Predict {conf_label:18s}: balanced acc = {Y_conf:.3f}")
        if Y_conf >= X_skill:
            print(f"\n  🚨 STRONG WARNING: the model tells '{conf_label}' apart as well as or")
            print("     better than SKILL → the skill result is NOT TRUSTWORTHY.")
            print("     The features are carrying recording-condition information.")
        elif Y_conf > 0.7:
            print(f"\n  ⚠️  The features DO carry '{conf_label}' information (>0.7). Interpret")
            print("     the skill result cautiously.")
        else:
            print(f"\n  → near 0.5: the features carry little '{conf_label}' information.")
    else:
        print(f"    Predict {conf_label}: (not run)")

    # (4c) Quality table per session — for viewing only
    quality_by_date(df)

    # ── OUTPUT ───────────────────────────────────────────────────────
    all_rows = pd.DataFrame(rows_skill + rows_conf)
    all_rows.to_csv(outdir / "results.csv", index=False)
    if not imp_skill.empty:
        imp_skill.to_csv(outdir / "feature_importance.csv", index=False)

    import joblib
    final = build_models()["RandomForest"].fit(
        df[feat_cols].to_numpy(dtype=float), df["skill_level"].to_numpy())
    joblib.dump(final, outdir / "models" / "spike_skill_model.joblib")
    (outdir / "models" / "feature_list.json").write_text(
        json.dumps(feat_cols, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "n_contacts": int(len(df)),
        "n_sessions": int(df["session_name"].nunique()),
        "n_players": int(n_players),
        "players_per_class": df.groupby("skill_level")["player_id"].nunique().to_dict(),
        "session_type_filter": args.session_type,
        "exclude_low_quality": bool(args.exclude_low_quality),
        "feature_set": args.feature_set,
        "n_features": len(feat_cols),
        "pooled_skill_level": pooled_skill,
        "confound_label": conf_label,
        "pooled_confound": pooled_conf,
        "verdict_skill": v_skill,
        "confound_best_bal_acc": None if not pooled_conf else float(Y_conf),
        "trustworthy": bool(n_players >= MIN_PLAYERS_TRUST),
    }
    (outdir / "models" / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    hr("OUTPUT")
    for f in ["results.csv", "feature_importance.csv",
              "models/spike_skill_model.joblib", "models/feature_list.json",
              "models/model_metrics.json"]:
        p = outdir / f
        print(f"    {'✅' if p.exists() else '❌'} {p}")
    print(f"\n  skill_level conclusion: {v_skill}")


if __name__ == "__main__":
    main()
