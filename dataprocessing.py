import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

from shareddata import shared_data


# =========================
# CONFIG
# =========================

TRAIL_LENGTH = 30
SMOOTHING = 0.8

# =========================
# GLOBALS
# =========================

smooth_shoulder = None
wrist_trail = []

# =========================
# GET LATEST CAMERA POSE
# =========================

def get_pose():

    if not shared_data.camera_buffer:
        return None

    return shared_data.camera_buffer[-1]


# =========================
# LIVE GRAPH
# =========================

def live_graph():

    global smooth_shoulder
    global wrist_trail

    plt.ion()

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')

    while (
    plt.fignum_exists(fig.number)
    and shared_data.program_running
    ):

        # =========================
        # WAIT UNTIL RECORDING
        # =========================

        if not shared_data.graph_running:

            plt.pause(0.05)
            continue

        pose = get_pose()

        if pose is None:

            plt.pause(0.01)
            continue

        # =========================
        # GET JOINTS
        # =========================

        shoulder = np.array(pose["shoulder"])

        elbow = np.array(pose["elbow"])

        wrist = np.array(pose["wrist"])

        # =========================
        # SHOULDER SMOOTHING
        # =========================

        if smooth_shoulder is None:

            smooth_shoulder = shoulder.copy()

        smooth_shoulder = (
            SMOOTHING * smooth_shoulder +
            (1 - SMOOTHING) * shoulder
        )

        # =========================
        # RELATIVE COORDS
        # =========================

        shoulder_rel = np.array([0, 0, 0])

        elbow_rel = elbow - smooth_shoulder

        wrist_rel = wrist - smooth_shoulder

        # =========================
        # SAVE WRIST TRAIL
        # =========================

        wrist_trail.append(wrist_rel.copy())

        if len(wrist_trail) > TRAIL_LENGTH:

            wrist_trail.pop(0)

        # =========================
        # ARM POINTS
        # =========================

        xs = [
            shoulder_rel[0],
            elbow_rel[0],
            wrist_rel[0]
        ]

        ys = [
            shoulder_rel[1],
            elbow_rel[1],
            wrist_rel[1]
        ]

        zs = [
            shoulder_rel[2],
            elbow_rel[2],
            wrist_rel[2]
        ]

        # =========================
        # CLEAR
        # =========================

        ax.clear()

        # =========================
        # DRAW JOINTS
        # =========================

        ax.scatter(
            xs,
            ys,
            zs,
            s=80
        )

        # =========================
        # DRAW BONES
        # =========================

        ax.plot(
            xs,
            ys,
            zs,
            linewidth=3
        )

        # =========================
        # DRAW TRAIL
        # =========================

        trail_x = [p[0] for p in wrist_trail]
        trail_y = [p[1] for p in wrist_trail]
        trail_z = [p[2] for p in wrist_trail]

        ax.plot(
            trail_x,
            trail_y,
            trail_z,
            alpha=0.5
        )

        # =========================
        # LABELS
        # =========================

        ax.set_title("Realtime Motion Capture")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # =========================
        # FIXED VIEW
        # =========================

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax.set_box_aspect([1, 1, 1])

        ax.view_init(
            elev=20,
            azim=45
        )

        plt.pause(0.01)

    plt.close("all")


# =========================
# SAVE REPLAY
# =========================

def save_replay():

    if shared_data.session_dir is None:

        print("No session directory")
        return

    csv_path = shared_data.session_dir / "camera_data.csv"

    if not csv_path.exists():

        print("camera_data.csv not found")
        return

    print("Loading camera CSV...")

    history = pd.read_csv(csv_path)

    if len(history) == 0:

        print("No motion data")
        return

    print(f"Frames loaded: {len(history)}")

    # IMPORTANT:
    # use NON GUI matplotlib objects
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FFMpegWriter

    # =========================
    # CREATE FIGURE
    # =========================

    fig = Figure(figsize=(8, 8))

    FigureCanvasAgg(fig)

    ax = fig.add_subplot(111, projection='3d')

    # =========================
    # OUTPUT FILE
    # =========================

    filename = shared_data.session_dir / "replay.mp4"

    print("Saving replay to:", filename)

    writer = FFMpegWriter(
        fps=30,
        bitrate=1800
    )

    # =========================
    # SAVE VIDEO
    # =========================

    with writer.saving(fig, str(filename), dpi=100):

        for frame in range(len(history)):

            ax.clear()

            row = history.iloc[frame]

            shoulder = np.array([
                row["shoulder_x"],
                row["shoulder_y"],
                row["shoulder_z"]
            ])

            elbow = np.array([
                row["elbow_x"],
                row["elbow_y"],
                row["elbow_z"]
            ])

            wrist = np.array([
                row["wrist_x"],
                row["wrist_y"],
                row["wrist_z"]
            ])

            # =========================
            # RELATIVE COORDS
            # =========================

            elbow_rel = elbow - shoulder
            wrist_rel = wrist - shoulder

            xs = [
                0,
                elbow_rel[0],
                wrist_rel[0]
            ]

            ys = [
                0,
                elbow_rel[1],
                wrist_rel[1]
            ]

            zs = [
                0,
                elbow_rel[2],
                wrist_rel[2]
            ]

            # =========================
            # DRAW ARM
            # =========================

            ax.scatter(
                xs,
                ys,
                zs,
                s=80
            )

            ax.plot(
                xs,
                ys,
                zs,
                linewidth=3
            )

            # =========================
            # WRIST TRAIL
            # =========================

            start = max(0, frame - TRAIL_LENGTH)

            trail = history.iloc[start:frame]

            trail_x = (
                trail["wrist_x"].values -
                trail["shoulder_x"].values
            )

            trail_y = (
                trail["wrist_y"].values -
                trail["shoulder_y"].values
            )

            trail_z = (
                trail["wrist_z"].values -
                trail["shoulder_z"].values
            )

            ax.plot(
                trail_x,
                trail_y,
                trail_z,
                alpha=0.5
            )

            # =========================
            # VIEW
            # =========================

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

            ax.set_box_aspect([1, 1, 1])

            ax.view_init(
                elev=20,
                azim=45
            )

            ax.set_title(f"Replay Frame {frame}")

            # render frame
            writer.grab_frame()

    print("Replay saved:", filename)