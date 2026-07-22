# FILE_MAP.md — File map of the ANALYSIS & MODELLING half

> **Scope:** the data-analysis and model-training half only.
> **Excludes** the data-collection group: `mainpy.py`, `cameraclient.py`,
> `imuclient.py`, `shareddata.py`.
>
> Updated: 2026-07-22. Written for someone who has never seen this repo.

> ### 🔄 UPDATE 2026-07-22 — English translation
> The entire codebase (comments, docstrings, prints, error messages, report UI
> labels) has been translated to English. Variable names, function names, CSV
> column names, file names, and CLI arguments are unchanged. `model_metrics.json`
> is now pure ASCII (its verdict text is English) — see item 10.

> ### 🔄 UPDATE 2026-07-20 — automated the recording → dataset flow
> - **`mainpy.py`** now asks for the session kind (professional/amateur/**test**)
>   before recording, and writes a **`session_info.json`** into the session folder.
>   Test sessions are named `TEST_...`. *(mainpy still belongs to the data-collection
>   group — mentioned here because it produces `session_info.json`, which the analysis
>   half reads.)*
> - **`src/promote_session.py`** (NEW) — one command instead of three manual steps:
>   copy a session from `recording/` into `Zipped data/`, auto-refuse test sessions,
>   then update the metadata. See Part 2 item 5b and Part 5.
> - **`prepare_dataset.py`** now **MERGES** instead of overwriting: re-running keeps
>   the hand-edited `good_recording` / `notes`, backs up a `.bak`, and only appends new rows.
> - **`metadata.csv` schema changed**: dropped `dominant_hand` / `device_wrist` /
>   `camera_view` (the project only records male right-handed athletes). `session_type`
>   defaults to **`unrated`** — assign primary/secondary yourself after reviewing the
>   video. See Part 4.
> - **`train_model.py`** added `--confound-label` (session_type/session_date/none),
>   a per-feature Spearman-correlation table against recording quality, and a
>   per-session quality table. See Part 2 item 5.

---

## PART 1 — DATA-FLOW DIAGRAM

```
  [SOURCE DATA — you record it yourself, no script generates it]
  ../Zipped data/                          ← lives OUTSIDE motioncapture/
   ├── Primary sessions/
   │    ├── Amatures/<player>_<ts>_primary/     ← 1 folder = 1 spike attempt
   │    │     ├── camera_data.csv
   │    │     ├── imu_data.csv
   │    │     └── annotated_video.mp4
   │    └── Proffesionals/...
   └── Secondary sessions/{Amatures,Proffesionals}/...
                    │
                    │  (branch A — data prep, OPTIONAL)
                    ├─────────────────────────────────────┐
                    │                                     ▼
                    │                       ▣ src/prepare_dataset.py
                    │                                     │ copy + rename
                    │                                     ▼
                    │                          ▢ prepared_dataset/
                    │                            ├── <attempt_id>_camera.csv  (×24)
                    │                            ├── <attempt_id>_imu.csv     (×24)
                    │                            ├── <attempt_id>_video.mp4   (×24)
                    │                            └── ▢ metadata.csv  ★
                    │                                  (script generates the skeleton,
                    │                                   YOU fill in a few columns)
                    │                                     │
                    │  (branch B — feature extraction)    │
                    ▼                                     │
        ▣ extraction.py  --root "../Zipped data"          │
                    │                                     │
                    ▼                                     │
             ▢ features.csv   (30 rows × 54 columns)      │
              1 row = 1 detected spike                    │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   ▼
                      ▣ src/train_model.py
                        (joins the 2 files on player+timestamp)
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
   ▢ results.csv        ▢ feature_importance.csv    ▢ models/
   ▢ results_feature_sets.csv                        ├── spike_skill_model.joblib
                                                     ├── feature_list.json
                                                     └── model_metrics.json


  [SEPARATE BRANCH — review one session by eye, NOT wired into the model flow]
  1 session folder ──▶ ▣ dataprocessing.py ──▶ ▢ motion_report.html
                                            ▶ ▢ graphs_only.html
                                            ▶ ▢ Open_Viewer.bat
                                            (written into the session folder itself)
```

**Legend**

| | |
|---|---|
| ▣ | **Script** — run directly with `python` |
| ▢ | **Data** — produced by a script |
| ★ | A file you **must edit by hand** after the script generates the skeleton |

**Key point:** `dataprocessing.py` is a **standalone** branch, only for reviewing a
session by eye. It does **not** produce `features.csv` and is **not** involved in
model training. The two files `extraction.py` and `dataprocessing.py` do not import
each other (verified).

---

## PART 2 — FILE TABLE

In flow order.

---

### 1. `src/prepare_dataset.py`

| | |
|---|---|
| **Path** | `motioncapture/src/prepare_dataset.py` |
| **Type** | Script run directly |
| **Author** | Claude, per your request |

**Job:** Scan `Zipped data`, copy each session's 3 files into a flat
`prepared_dataset/` folder with a uniform prefix, and generate the `metadata.csv`
skeleton. **Never modifies or deletes the source files.**

- **Reads:** `../Zipped data/**` (hard-coded path in the code, see `SOURCE_DIR`)
- **Writes:** `prepared_dataset/` (72 data files + `metadata.csv`)

**Command:**
```bash
python src/prepare_dataset.py
```
No command-line arguments. To change the paths you must edit `SOURCE_DIR` /
`OUTPUT_DIR` in the code.

**Safe to delete?** Yes, if `prepared_dataset/metadata.csv` already exists and you
have filled it in. Losing this script means recreating `metadata.csv` by hand.

> ⚠️ **Important warning:** `prepared_dataset/` holds **flat prefixed files**
> (`professional_primary_Hieu_..._camera.csv`), whereas `extraction.py` needs a
> **session folder** containing a file named `camera_data.csv`. That means you
> **cannot run `extraction.py` directly on `prepared_dataset/`**. Right now
> `prepared_dataset/` is only used to produce `metadata.csv`; feature extraction
> reads straight from `Zipped data`.

---

### 2. `prepared_dataset/metadata.csv` ★

| | |
|---|---|
| **Path** | `motioncapture/prepared_dataset/metadata.csv` |
| **Type** | Input data — **script generates the skeleton, you fill it in** |
| **Author** | Generated by `src/prepare_dataset.py`; some columns left blank for you |

**Job:** The **single source of truth** deciding which sessions are used for
research. `train_model.py` only accepts rows present in this file — any session not
listed here is dropped, even if it exists in `features.csv`.

- Currently **24 rows** (24 sessions)
- Per-column detail: see **Part 4**

**Safe to delete?** **NO.** Losing this file means `train_model.py` cannot run
(the `--metadata` argument is required). Your hand-filled columns would be lost too.

---

### 3. `extraction.py`

| | |
|---|---|
| **Path** | `motioncapture/extraction.py` — **at the root, NOT in `src/`** |
| **Type** | Script run directly (also importable to reuse its functions) |
| **Author** | Originally yours, revised by Claude over several rounds |

**Job:** Read each session's raw data (camera + IMU), automatically find the
ball-contact moment, split the spike into 3 phases, then compute the numbers that
describe the spike. This is the **longest and most important** file of the analysis
half (~62 KB).

- **Reads:** `camera_data.csv` + `imu_data.csv` in each session folder
- **Writes:** a single CSV file (path set by `--output`)

**Three run modes:**

```bash
# (a) RECURSIVELY scan a nested folder tree — USE THIS MODE for the real dataset
python extraction.py --root "../Zipped data" --output features.csv --quiet

# (b) FLAT scan of one folder (all direct sub-folders)
python extraction.py recording/ --all --output features.csv

# (c) One session, print detail to the screen, no file saved
python extraction.py recording/Hieu_20260706_170956_primary
```

**Safe to delete?** **NO.** Losing this file means `features.csv` cannot be regenerated.

> ⚠️ Using mode (b) on `recording/` would **miss 4 sessions** (3 of Hoang, 1 of
> Hieu) because they only exist in `Zipped data`. Missing Hoang = losing one of the
> two professional players = `train_model.py` will stop. **Always use mode (a).**

---

### 4. `features.csv`

| | |
|---|---|
| **Path** | `motioncapture/features.csv` (default; change with `--output`) |
| **Type** | Output data — **auto-generated** |
| **Author** | `extraction.py` |

**Job:** The main data table. **One row = one detected spike** (not one session — a
session can have several spikes or none).

- Currently **30 rows × 54 columns**, from 21/24 sessions
  (3 sessions produced no spike so they don't appear)
- Per-column detail: see **Part 3**

**Safe to delete?** Yes — re-run the command in item 3 to get it back. ~2–3 minutes.

---

### 5. `src/train_model.py`

| | |
|---|---|
| **Path** | `motioncapture/src/train_model.py` |
| **Type** | Script run directly |
| **Author** | Claude, per your request |

**Job:** Join `features.csv` with `metadata.csv`, check for things that could bias
the result, then train and evaluate 3 professional/amateur classifiers. Split the
data **by player** (a player never appears in both the training and test set).

- **Reads:** `features.csv`, `metadata.csv`
- **Writes:** `results.csv`, `feature_importance.csv`, `models/` (3 files),
  and `results_feature_sets.csv` (compare mode only)

**Commands:**
```bash
# Standard run
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv

# Compare 4 feature sets to trace the confound (does NOT save a model)
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv \
       --compare-feature-sets

# Use IMU features only
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv \
       --feature-set imu_pure

# Drop low-quality spikes  ⚠️ currently filters down to 0 samples → the script stops
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv \
       --exclude-low-quality
```

**Safe to delete?** **NO.** Losing this file loses all training and confound checks.

---

### 6. `results.csv`

| | |
|---|---|
| **Path** | `motioncapture/results.csv` |
| **Type** | Output data — auto-generated |
| **Author** | `src/train_model.py` (standard run) |

**Job:** Detailed results per split. Columns:
`task, model, fold, test_player, n_train, n_test, n_classes_test,
balanced_accuracy, f1_macro`.

- Currently **24 rows** = 2 tasks × 3 models × 4 splits

**Safe to delete?** Yes — re-run `train_model.py` (a few seconds).

---

### 7. `results_feature_sets.csv`

| | |
|---|---|
| **Path** | `motioncapture/results_feature_sets.csv` |
| **Type** | Output data — auto-generated |
| **Author** | `src/train_model.py --compare-feature-sets` |

**Job:** Like `results.csv` but for **all 4 feature sets** (`all`, `imu_pure`,
`imu_phase`, `camera_only`). The `task` column looks like `skill_level::imu_pure`
to disambiguate.

- Currently **96 rows** = 4 sets × 2 tasks × 3 models × 4 splits

**Safe to delete?** Yes — re-run the compare mode (a few seconds).

---

### 8. `feature_importance.csv`

| | |
|---|---|
| **Path** | `motioncapture/feature_importance.csv` |
| **Type** | Output data — auto-generated |
| **Author** | `src/train_model.py` (standard run) |

**Job:** RandomForest feature importance, averaged over the 4 splits. Columns:
`feature, importance`. **37 rows**, sorted descending.

**Safe to delete?** Yes — re-run `train_model.py`.

---

### 9. `models/feature_list.json`

| | |
|---|---|
| **Path** | `motioncapture/models/feature_list.json` |
| **Type** | Output data — auto-generated |
| **Author** | `src/train_model.py` |

**Job:** The list of **37 column names in the exact order** the model was trained
on. Needed for the future prediction step: new data must be arranged in this order
for the model to understand it.

**Safe to delete?** Yes, but delete it together with `spike_skill_model.joblib` —
the two files are a pair; a mismatch produces silently wrong predictions.

---

### 10. `models/model_metrics.json`

| | |
|---|---|
| **Path** | `motioncapture/models/model_metrics.json` |
| **Type** | Output data — auto-generated |
| **Author** | `src/train_model.py` |

**Job:** A summary of the run: number of spikes, sessions, players, the parameters
used, the pooled results of all 3 models for both tasks, and a text verdict.

Current values: `n_contacts=30`, `n_sessions=21`, `n_players=4`,
`players_per_class={amateur:2, professional:2}`, `feature_set=all`,
`n_features=37`, `trustworthy=false`.

> ✅ **Encoding note:** since the English translation, this file is **pure ASCII**
> (the verdict text is English). It no longer requires `encoding="utf-8"` to read on
> Windows — plain `open(...)` works. (Before the translation it contained Vietnamese
> and did require utf-8.)

**Safe to delete?** Yes — re-run `train_model.py`.

---

### 11. `models/spike_skill_model.joblib`

| | |
|---|---|
| **Path** | `motioncapture/models/spike_skill_model.joblib` |
| **Type** | Output data — auto-generated (~327 KB) |
| **Author** | `src/train_model.py` |

**Job:** The RandomForest model trained on **all** 30 spikes (not a single split's
model). Used for the future new-spike prediction step.

> ⚠️ **This model is currently NOT usable for real prediction.** The evaluation
> shows it cannot tell skill levels apart (see Part 6). The file exists to complete
> the pipeline, not because it works.

**Safe to delete?** Yes — re-run `train_model.py`. Remember to delete
`feature_list.json` too so the two files stay in sync.

---

### 12. `dataprocessing.py`

| | |
|---|---|
| **Path** | `motioncapture/dataprocessing.py` (~61 KB) |
| **Type** | Script run directly |
| **Author** | Originally yours, Claude revised the video-sync part |

**Job:** Review **one** session by eye: open a window with the video on the left and
the elbow-angle + acceleration charts on the right, running in sync. Used to sanity-
check the data, **not part of model training**.

- **Reads:** one session folder, or any video file (`.mp4/.avi/.mov`)
- **Writes (into the session folder itself):** `motion_report.html`, `graphs_only.html`,
  `Open_Viewer.bat`

**Commands:**
```bash
python dataprocessing.py recording/Hieu_20260706_170956_primary
python dataprocessing.py recording/<session> --no-viewer     # export HTML only
python dataprocessing.py spike.mp4 --side RIGHT              # analyze a standalone video
```

**Safe to delete?** Yes **if you don't need the review tool**. The model-training
flow doesn't depend on this file. But losing it means losing the only way to check
the data by eye.

---

### 13. Auxiliary files (not part of the main flow)

| File | Type | Notes |
|---|---|---|
| `README.md` | Docs | Describes the whole system. **Updated** to cover both halves (collection + extraction + training). |
| `pyproject.toml` | Config | Library list. `scikit-learn` + `joblib` were added for `train_model.py`. |
| `uv.lock` | Auto-generated | Library version lock. **Do not edit by hand.** |
| `recording/` | Data | ~157 session folders, **mostly test/throwaway**. Not the research dataset. |
| `models/` | Folder | Contains the 3 output files in items 9–11. |

---

## PART 3 — COLUMNS IN `features.csv`

**54 columns.** The "In model?" column reflects the default mode (`--feature-set all`):
**37 columns** used, 17 dropped.

### 3.1 — Identifier columns (2) — NOT in the model

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `session_name` | text | Session folder name, e.g. `Hieu_20260706_170956_primary`. The key for joining with metadata. | ❌ Identifier, not a feature |
| `contact_frame` | index | Frame index where ball contact occurs | ❌ Internal technical index |

### 3.2 — Quality-flag columns (7) — NOT in the model, used for FILTERING

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `has_imu` | 0/1 | Whether the session has IMU data | ❌ Technical flag |
| `imu_timebase` | text | Which time axis is in use (`arduino_time`/`pc_time`) | ❌ String type |
| `clip_fraction` | ratio 0–1 | **% of accel samples hitting the ±4G ceiling** over the whole session | ❌ Quality flag |
| `peak_is_clipped` | 0/1 | Whether the sample at the exact contact moment is clipped | ❌ Quality flag |
| `contact_near_gap` | 0/1 | Whether the spike sits next to a data gap | ❌ Quality flag |
| `sync_quality_ms` | milliseconds | Estimated error when aligning the camera–IMU clocks. Lower is better. | ❌ Quality flag |
| `sync_lowlat_fraction` | ratio 0–1 | % of samples with transmission latency ≤10ms | ❌ Quality flag |
| `n_contacts_rejected` | count | Number of spikes rejected because camera and IMU don't coincide in time | ❌ Quality flag |

### 3.3 — `cam_*` columns (10) — ✅ in the model

Computed from the camera via MediaPipe. **A 2D projection, so it depends on camera angle.**

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `cam_angle_at_contact_deg` | degrees | Elbow angle at ball contact | ✅ |
| `cam_angle_max_cocking_deg` | degrees | Max elbow angle during cocking | ✅ |
| `cam_angle_min_acceleration_deg` | degrees | Min elbow angle during the arm swing | ✅ |
| `cam_angle_rom_deg` | degrees | Elbow range of motion (max − min) over the session | ✅ |
| `cam_angle_rate_cocking_dps` | deg/sec | Rate of angle change during cocking | ✅ |
| `cam_angle_rate_acceleration_dps` | deg/sec | Rate of angle change during the swing (negative = snapping down) | ✅ |
| `cam_angle_rate_followthrough_dps` | deg/sec | Rate of angle change after contact | ✅ |
| `cam_wrist_path_length_norm` | arm lengths | Total distance the wrist travels during the spike | ✅ |
| `cam_wrist_displacement_norm` | arm lengths | Straight-line distance from cocking start to contact | ✅ |
| `cam_wrist_peak_speed_norm` | arm lengths/sec | Max wrist speed | ✅ |

> The three `_norm` columns are divided by arm length to remove the effect of the
> player standing near/far from the camera.
>
> ⚠️ All 10 columns use 3D coordinates, where **MediaPipe's z (depth) axis is a rough,
> unreliable estimate**. Measured: the elbow angle computed with z vs with x,y only
> differs by 21–42° on average, up to 143° at the extreme.

### 3.4 — `phase_*` columns (6) — ✅ in the model

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `phase_cocking_duration_s` | seconds | Duration of the cocking phase | ✅ |
| `phase_acceleration_duration_s` | seconds | Duration of the arm-swing phase | ✅ |
| `phase_followthrough_duration_s` | seconds | Duration of the follow-through phase after contact | ✅ |
| `phase_total_duration_s` | seconds | Sum of the 3 phases | ✅ |
| `phase_cocking_pct` | % | Cocking phase as a % of total time | ✅ |
| `phase_acceleration_pct` | % | Swing phase as a % of total time | ✅ |

> ⚠️ **`phase_*` is NOT fully independent of the camera angle.** The cocking-phase
> end boundary is set by the elbow-angle maximum — and the elbow angle comes from the
> camera. So `phase_cocking_duration_s` and `phase_acceleration_duration_s` depend
> **indirectly** on the camera angle. Only `phase_followthrough_duration_s` is nearly independent.

### 3.5 — `imu_*` columns (21) — ✅ in the model

From the arm-worn sensor. In principle independent of the camera position.

**Acceleration group — ⚠️ NOT TRUSTWORTHY when `clip_fraction` is high:**

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `imu_peak_acc_g` | G | Max acceleration around contact | ✅ ⚠️ **clipped** |
| `imu_acc_at_contact` | G | Acceleration at the exact contact moment | ✅ ⚠️ **clipped** |
| `imu_ratio_gyro_acc` | ratio | Rotation divided by acceleration at contact | ✅ ⚠️ **denominator clipped** |
| `imu_time_to_peak_acc_s` | seconds | Time from window start to the acceleration peak | ✅ ⚠️ **peak clipped** |
| `imu_acc_jerk_mean_abs` | G/sec | Mean jerk (derivative of acceleration) | ✅ ⚠️ |
| `imu_acc_jerk_max_abs` | G/sec | Max jerk | ✅ ⚠️ |
| `imu_acc_jerk_rms` | G/sec | RMS jerk | ✅ ⚠️ |
| `imu_acc_jerk_dimensionless` | dimensionless | Jerk normalized by duration and amplitude | ✅ ⚠️ |

> **Why untrustworthy:** the LSM9DS1 sensor hard-clips at ±4G per axis, and the
> Arduino library can't change it. Measured: **69% of spikes have the sample at
> contact clipped**. Hard and very-hard spikes read the same ~6.93G, so this feature
> **compresses away** the difference between skill groups. Differentiating the
> flat-topped signal also produces a very large **fake** jerk.

**Gyroscope group — ✅ CLIP-SAFE, trustworthy:**

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `imu_peak_gyro_dps` | deg/sec | Max rotation speed around contact | ✅ trustworthy |
| `imu_gyro_at_contact` | deg/sec | Rotation speed at contact | ✅ trustworthy |
| `imu_gyro_impulse_duration_s` | seconds | Time the rotation speed stays above 50% of peak ("snap duration") | ✅ trustworthy |
| `imu_gyro_jerk_mean_abs` | deg/sec² | Mean rotational jerk | ✅ trustworthy |
| `imu_gyro_jerk_max_abs` | deg/sec² | Max rotational jerk | ✅ trustworthy |
| `imu_gyro_jerk_rms` | deg/sec² | RMS rotational jerk | ✅ trustworthy |
| `imu_gyro_jerk_dimensionless` | dimensionless | Normalized rotational jerk | ✅ trustworthy |
| `imu_gyro_jerk_rms_cocking` | deg/sec² | Rotational jerk in the cocking phase only | ✅ trustworthy |
| `imu_gyro_jerk_rms_acceleration` | deg/sec² | Rotational jerk in the swing phase only | ✅ trustworthy |
| `imu_gyro_jerk_rms_followthrough` | deg/sec² | Rotational jerk in the follow-through phase only | ✅ trustworthy |

> The gyroscope's ceiling is ±2000 deg/sec, and only **0.23% of samples at most** hit
> it → effectively not clipped.

**Other group:**

| Column | Unit | Meaning | In model? |
|---|---|---|---|
| `imu_sample_rate` | Hz | The session's actual average sample rate (22–68Hz due to packet loss) | ✅ ⚠️ see note |
| `contact_ts_offset_s` | seconds | At what second the spike occurs, measured from the session start | ✅ ⚠️ see note |
| `ipf_confidence` | 0–1 | Confidence of the contact detection | ✅ ⚠️ tainted by clipped acceleration |

> ⚠️ `imu_sample_rate` and `contact_ts_offset_s` describe **recording conditions**,
> not spike technique. They are still fed to the model in `all` mode — consider
> dropping them (see Part 6).
>
> ⚠️ `ipf_confidence` is actually an IMU quantity but has **no `imu_` prefix**, so
> `--feature-set imu_pure` drops it by mistake. Same for `contact_ts_offset_s`. These
> two columns only appear in `all`.

### 3.6 — `cam_vis_*` columns (7) — ❌ NEVER put in the model

| Column | Unit | Meaning |
|---|---|---|
| `cam_vis_shoulder_mean` | 0–1 | Mean confidence of detecting the shoulder joint |
| `cam_vis_elbow_mean` | 0–1 | Mean confidence of the elbow joint |
| `cam_vis_wrist_mean` | 0–1 | Mean confidence of the wrist joint |
| `cam_vis_wrist_pct_below_05` | % | % of frames where the wrist is poorly detected (<0.5) |
| `cam_vis_min_of_three` | 0–1 | Minimum of the three values above (the weakest point) |
| `cam_vis_wrist_mean_window` | 0–1 | As above but only during the spike |
| `cam_vis_wrist_pct_below_05_window` | % | As above but only during the spike |

> ❌ **These are DATA QUALITY indicators, not technique features.** They reflect
> recording conditions (lighting, camera angle, occlusion), not skill. Measured:
> professional group **0.864** vs amateur **0.929** — skewed **opposite** to skill.
> Feeding them in teaches the model to guess "recording conditions" and report a
> falsely high score. The script drops them automatically via the `cam_vis_` prefix rule.

---

## PART 4 — COLUMNS IN `metadata.csv`

**New schema: 10 columns** (dropped `dominant_hand` / `device_wrist` / `camera_view`
— the project only records male right-handed athletes). If those 3 legacy columns
still exist, `prepare_dataset.py` **keeps them**, it just doesn't generate new ones.

| Column | Who fills it | Meaning | Required? |
|---|---|---|---|
| `attempt_id` | 🤖 script | Identifier: `{skill}_{angle}_{player}_{timestamp}` | ✅ Join key |
| `player_id` | 🤖 script (from `session_info.json`) | Player name | ✅ Used to split the data |
| `skill_level` | 🤖 script | `professional` / `amateur`, **the label to predict**. If it can't be inferred it's left blank and the script **lists it for you to fill in**. | ✅ |
| `session_type` | 🤖 script → ✍️ **you edit** | Defaults to **`unrated`**. After reviewing the video, change to `primary`/`secondary` if needed. | ✅ (used as the confound label) |
| `session_date` | 🤖 script | Recording date, parsed from the timestamp in the name | ❌ (only for the per-session quality table) |
| `imu_path` | 🤖 script | IMU file name in `prepared_dataset/` | ❌ Not used in training |
| `camera_path` | 🤖 script | Camera file name | ❌ |
| `video_path` | 🤖 script | Video file name | ❌ |
| `good_recording` | 🤖 default `yes` → ✍️ **you edit** | Change to `no` to drop a broken session. **Merge mode KEEPS** this value on re-run. | ✅ |
| `notes` | ✍️ **you fill in** | Free-text notes. Merge keeps it. | ❌ |

> **JUDGMENT vs AUTOMATION:** `session_type`, `good_recording`, `notes` require you
> to watch the video and decide — the script does **not** automate this, it only sets
> defaults. Re-running `prepare_dataset.py`/`promote_session.py` does **not erase** your edits.

### How the join key works

This is the easiest place to get confused: **the two files use different name formats.**

```
metadata.csv  →  attempt_id    = "professional_primary_Hieu_20260706_170956"
                                  └─skill──┘└─angle─┘└player┘└─timestamp──┘

features.csv  →  session_name  = "Hieu_20260706_170956_primary"
                                  └player┘└─timestamp──┘└─angle─┘
```

A direct name join matches **0 rows**. So `train_model.py` **normalizes both to the
same key**:

```
key = "{player name lowercased}_{timestamp}"

"professional_primary_Hieu_20260706_170956"  →  "hieu_20260706_170956"
"Hieu_20260706_170956_primary"               →  "hieu_20260706_170956"   ✅ match
```

The (player, timestamp) pair is **unique** across all 24 sessions, so it's enough as
a key. The script also accepts a `metadata.csv` that already has a `session_name`
column (it uses that directly when present).

---

## PART 5 — FULL RUN WORKFLOW

Run every command from the `motioncapture/` folder.

---

### Step 0 — Set up the environment (once)

```bash
uv sync
```
- **Time:** 1–3 minutes the first time, then a few seconds
- **Success check:** `./.venv/Scripts/python.exe -c "import sklearn, cv2, mediapipe; print('OK')"`

---

### Step 1 — Add a new session to the dataset (each time you record more)

```bash
python src/promote_session.py recording/<session_name>
```
- **Time:** a few seconds (no video copy if the session already exists)
- **What it does:** copy the session into `Zipped data/{Proffesionals|Amatures}/`,
  **auto-refuse test sessions**, then call `prepare_dataset.py` to update
  `metadata.csv` (merge).
- **Success check:** prints `✅ Copied to: ...` and `MERGE: kept N existing, added M new`
- Old session with no `session_info.json`: add `--skill professional|amateur`.
- Preview without touching anything: add `--dry-run`.

> After this step, **open `metadata.csv` and review it by eye** (the JUDGMENT part,
> not automated): set `good_recording=no` for broken sessions, fill in `session_type`
> (primary/secondary) after watching the video, add `notes`. Re-running
> `promote_session.py`/`prepare_dataset.py` **does NOT erase** these edits (merge mode).

> To just re-scan the whole `Zipped data` without promoting a specific session:
> `python src/prepare_dataset.py` (also merges, backs up a `.bak`).

---

### Step 2 — Extract features

```bash
python extraction.py --root "../Zipped data" --output features.csv --quiet
```
- **Time:** **~2 seconds** for ~24 sessions (only reads CSVs, no MediaPipe)
- **Produces:** `features.csv`
- **Success check:** the folder-group table + the final line
  `... contact events from N/M session`

> ⚠️ **You must use `--root`.** Running `--all` on `recording/` would miss player Hoang.

---

### Step 3 — Train and evaluate

```bash
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv
```
- **Time:** 5–15 seconds
- **Produces:** `results.csv`, `feature_importance.csv`, `models/` (3 files)
- **Success check:** the line `✅ USED FOR TRAINING: 30 contacts from 21 session(s), 4 player(s)`
  — **the "4 players" number is the most important**; if it says 3, the data source is wrong

---

### Step 4 — Confound check (recommended each time you add data)

```bash
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv \
       --compare-feature-sets
```
- **Time:** 20–40 seconds (runs 4 feature sets)
- **Produces:** `results_feature_sets.csv`
- **Success check:** prints the 4-row summary table + the "INTERPRETATION" section +
  the "SIDE CHECK" table
- **What to read:** the `session_type` column. If it is **higher** than the
  `skill_level` column, the model tells *camera angle / recording session* apart
  better than *skill* → the skill result is not trustworthy.

---

## PART 6 — CURRENT STATUS

### 6.1 — Status table

| File | Status | Known issue |
|---|---|---|
| `src/prepare_dataset.py` | ✅ Works well | Source/dest paths hard-coded, no CLI arguments |
| `extraction.py` | ⚠️ Works, has known issues | See 6.2 |
| `src/train_model.py` | ⚠️ Works, has known issues | See 6.3 |
| `dataprocessing.py` | ✅ Works well | The video–graph sync after the last fix hasn't been verified by eye; the math was checked but no one has watched it for real |
| `predict_one_spike.py` (step 13) | ❌ **Does not exist yet** | See 6.4 |
| Wiring prediction into the viewer (step 14) | ❌ **Does not exist yet** | See 6.4 |
| `README.md` | ✅ Updated | Now covers both halves (collection + extraction + training) |

### 6.2 — `extraction.py`: known issues

1. **Cannot extract features from `prepared_dataset/`.** That folder holds flat
   prefixed files, while the script needs a folder containing a file named
   `camera_data.csv`. It currently reads straight from `Zipped data`.
2. **Three contact-detection constants are still counted in samples, not seconds**
   (`IPF_WINDOW=5`, `MIN_PEAK_DIST=50`). Their physical meaning shifts per session
   because the sample rate varies 22–68Hz. *Reported, you haven't asked to fix it.*
3. **MediaPipe's z axis is unreliable** but is still used for every camera feature.
   There is no 2D-only option to cross-check. *Reported, not fixed.*
4. **5 sessions have a low detection rate** (53–66%, gaps up to 140 frames) which
   skews the phase-segmentation window. Worse: all 5 are professional → a systematic
   skew between the two groups. *Reported, not fixed.*

### 6.3 — `src/train_model.py`: known issues

1. **The model cannot tell skill apart.** Pooled result: `skill_level = 0.095`
   while random guessing scores `0.325`. The confusion matrix shows the model
   **inverted**: it labels all 9 amateur spikes as professional. The cause is almost
   certainly that each class has only 2 players → the model learns individual traits.
2. **Serious confound, and it is NOT from the camera as once assumed.** The same
   feature set predicts `session_type` at **0.790**. Notably: `imu_pure` also hits
   **0.790** (equal to the full set), while `camera_only` is only **0.642**. The
   evidence points to **recording conditions differing between sessions**:
   `sync_quality_ms` differs by 46.6% between primary and secondary. Dropping camera
   features does **not** fix it.
3. **`--exclude-low-quality` is currently unusable** — after filtering, 0 samples
   remain, because 100% of spikes have `sync_quality_ms > 50`. The script stops with
   a message (by design).
4. **`--session-type primary` / `secondary` both fail to run** — after filtering,
   each class has only 1 player left. The script stops with a message (by design).
5. **Two columns that maybe should be dropped from the model:** `imu_sample_rate`
   and `contact_ts_offset_s` describe recording conditions, not technique, but are
   still fed in by default. *Undecided.*
6. **`ipf_confidence` is dropped by mistake under `--feature-set imu_pure`** because
   it lacks the `imu_` prefix, even though it is an IMU quantity. The cleanest fix is
   to rename the column in `extraction.py`. *Undecided.*

### 6.4 — What does not exist yet (steps 13–14 of the plan)

| Item | Status |
|---|---|
| `predict_one_spike.py` (step 13) | ❌ No file yet. Will need to read `models/spike_skill_model.joblib` + `models/feature_list.json` (already present, correct column order). |
| Wiring prediction into the viewer (step 14) | ❌ Not present. `dataprocessing.py` currently reads nothing from `models/`. |

> ⚠️ Before doing steps 13–14, know this: **the current model cannot tell skill
> apart** (item 6.3). Building a prediction tool on top of it would produce
> meaningless output. Prioritize recording more players and handling the
> recording-conditions confound first.

### 6.5 — Cleanup candidates

| Item | Assessment |
|---|---|
| `__pycache__/` | ✅ **Deletable now.** Python regenerates it. Already in `.gitignore`. |
| `recording/` (157 folders) | ⚠️ **Mostly test junk** — names with `test_`, `error`, `potential`, or all digits. Only 20/24 research sessions are here, and **4 sessions only exist in `Zipped data`**. Deletable once you're sure `Zipped data` is complete, but **back it up first**. |
| `recording/**/motion_report.html`, `graphs_only.html`, `Open_Viewer.bat` | ✅ **Deletable.** Produced by `dataprocessing.py`, regenerate by re-running. |
| 3 sessions with an empty `camera_data.csv` (0 bytes) | ⚠️ In `recording/`, they cause an error if you run `--all` on that folder. No effect on the `--root` flow. |
| `prepared_dataset/*.mp4` (24 files; the whole folder is 368 MB) | ⚠️ **No analysis script reads them.** Only for reviewing by eye. Deletable if you need space — the originals are still in `Zipped data`. |
| `features.csv`, `results*.csv`, `feature_importance.csv` | ✅ Deletable, regenerate via Steps 2–4. |

### 6.6 — Version-control warning

**The most important analysis files are currently NOT tracked by git:**

```
?? extraction.py       ← never committed
?? src/                ← never committed (train_model.py, prepare_dataset.py)
?? models/             ← never committed
```

They are **not blocked by `.gitignore`** — they simply have never been `git add`-ed.
The repo currently only tracks the data-collection group + `dataprocessing.py`.

> **A real incident already happened:** `extraction.py` was once fully overwritten,
> losing all changes (the folder is inside OneDrive; the editor held an old copy in
> memory and saved over it). Because it wasn't in git, it couldn't be recovered and
> had to be redone from scratch.
>
> **Recommendation:** `git add extraction.py src/ && git commit`.
> `models/` and `*.csv` are optional — they regenerate, and `.gitignore` already
> blocks `*.csv`.
