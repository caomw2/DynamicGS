# This script is based on an original implementation by True Price.
# Created by liminghao
import argparse
from pathlib import Path
import sqlite3
import sys

import numpy as np


IS_PYTHON3 = sys.version_info[0] >= 3

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "FULL_OPENCV": 5,
    "SIMPLE_RADIAL_FISHEYE": 6,
    "RADIAL_FISHEYE": 7,
    "OPENCV_FISHEYE": 8,
    "FOV": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def array_to_blob(array):
    array = np.asarray(array)
    if IS_PYTHON3:
        return array.tobytes()
    return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id),
        )
        return cursor.lastrowid


def parse_camera_file(txt_path):
    cameras = []
    with open(txt_path, "r") as cam_file:
        for line in cam_file:
            if not line or line.startswith("#"):
                continue

            values = line.split()
            cameras.append(
                {
                    "camera_id": int(values[0]),
                    "model": CAMERA_MODEL_IDS[values[1]],
                    "width": int(values[2]),
                    "height": int(values[3]),
                    "params": np.asarray(values[4:12], dtype=np.float64),
                }
            )
    return cameras


def camTodatabase():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="database.db")
    parser.add_argument("--txt_path", type=str, default="colmap/sparse_cameras.txt")
    args = parser.parse_args()

    database_path = Path(args.database_path)
    txt_path = Path(args.txt_path)

    if not database_path.exists():
        print("ERROR: database path dosen't exist -- please check database.db.")
        return

    db = COLMAPDatabase.connect(str(database_path))
    cameras = parse_camera_file(txt_path)

    for camera in cameras:
        db.update_camera(
            camera["model"],
            camera["width"],
            camera["height"],
            camera["params"],
            camera["camera_id"],
        )

    db.commit()

    rows = db.execute("SELECT * FROM cameras")
    for expected in cameras:
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)

        assert camera_id == expected["camera_id"]
        assert model == expected["model"]
        assert width == expected["width"]
        assert height == expected["height"]
        assert np.allclose(params, expected["params"])

    db.close()


if __name__ == "__main__":
    camTodatabase()
