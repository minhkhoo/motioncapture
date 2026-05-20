import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from shareddata import shared_data


# =========================
# GET LIVE CAMERA POSE
# =========================

def get_camera_pose():

    if not shared_data.camera_buffer:
        return None

    return shared_data.camera_buffer[-1]


# =========================
# LIVE 3D GRAPH
# =========================

def live_graph(stop_event=None):

    plt.ion()

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')

    while True:

        # stop graph thread
        if stop_event and stop_event.is_set():
            break

        pose = get_camera_pose()

        if pose is None:
            time.sleep(0.01)
            continue

        shoulder = pose["shoulder"]
        elbow = pose["elbow"]
        wrist = pose["wrist"]

        # =========================
        # ARM POINTS
        # =========================

        xs = [
            shoulder[0],
            elbow[0],
            wrist[0]
        ]

        ys = [
            shoulder[1],
            elbow[1],
            wrist[1]
        ]

        zs = [
            shoulder[2],
            elbow[2],
            wrist[2]
        ]

        # =========================
        # DRAW
        # =========================

        ax.clear()

        # joints
        ax.scatter(
            xs,
            ys,
            zs,
            s=80
        )

        # bones
        ax.plot(
            xs,
            ys,
            zs,
            linewidth=3
        )

        # =========================
        # LABELS
        # =========================

        ax.set_title("Realtime Arm Motion")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # =========================
        # BETTER VIEW
        # =========================

        ax.invert_yaxis()

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-1, 1)

        ax.set_box_aspect([1, 1, 1])

        ax.view_init(
            elev=20,
            azim=45
        )

        plt.pause(0.01)

    plt.close(fig)


# =========================
# MAIN
# =========================

def main():

    live_graph()


if __name__ == "__main__":

    main()