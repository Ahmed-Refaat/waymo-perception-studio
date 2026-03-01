#!/usr/bin/env python3
"""
Generate mock Waymo v2.0 Parquet fixture files for unit tests.

Usage:
    python scripts/generate_fixtures.py

Output:
    src/__fixtures__/mock_segment_0000/*.parquet

Deterministic: uses np.random.seed(42) for reproducibility.
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
SEGMENT = "mock_segment_0000"
NUM_FRAMES = 199
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "__fixtures__", SEGMENT)

# Range image dimensions (small for git-friendly fixtures)
SENSOR_DIMS = {
    1: (8, 100),   # TOP
    2: (8, 50),    # FRONT
    3: (4, 20),    # SIDE_LEFT
    4: (4, 20),    # SIDE_RIGHT
    5: (4, 20),    # REAR
}
VALID_RATIO = 0.88
OBJECTS_PER_FRAME = 75

np.random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# Shared timestamps (microseconds, monotonically increasing)
ts_base = 1_000_000_000_000
timestamps = [ts_base + i * 100_000 for i in range(NUM_FRAMES)]  # 100ms apart


def make_identity():
    """Row-major 4×4 identity."""
    return [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]


def make_vehicle_pose(frame_idx):
    """Straight-line drive: Tx increases 0.1m per frame."""
    tx = frame_idx * 0.1
    return [1, 0, 0, tx,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]


# ─── vehicle_pose ───────────────────────────────────────────────────────────

def gen_vehicle_pose():
    rows = {
        "key.segment_context_name": [SEGMENT] * NUM_FRAMES,
        "key.frame_timestamp_micros": timestamps,
        "[VehiclePoseComponent].world_from_vehicle.transform": [
            make_vehicle_pose(i) for i in range(NUM_FRAMES)
        ],
    }
    table = pa.table({
        k: pa.array(v, type=pa.list_(pa.float64(), 16)) if "transform" in k
           else pa.array(v)
        for k, v in rows.items()
    })
    pq.write_table(table, os.path.join(OUT_DIR, "vehicle_pose.parquet"))
    print(f"  vehicle_pose: {NUM_FRAMES} rows")


# ─── lidar_calibration ──────────────────────────────────────────────────────

def gen_lidar_calibration():
    laser_names = [1, 2, 3, 4, 5]
    top_height = SENSOR_DIMS[1][0]  # 8

    # TOP: non-uniform inclinations (ascending min→max, like real Waymo storage)
    inc_max, inc_min = 0.04, -0.29
    top_inclinations = np.linspace(inc_min, inc_max, top_height).tolist()

    extrinsics = []
    inc_values = []
    inc_mins = []
    inc_maxs = []

    for ln in laser_names:
        # Simple extrinsic: identity with z-offset (roof-mounted)
        ext = list(make_identity())
        ext[11] = 1.8  # Tz = 1.8m above ground
        extrinsics.append(ext)

        if ln == 1:  # TOP: explicit inclination values
            inc_values.append(top_inclinations)
        else:
            inc_values.append(None)
        inc_mins.append(inc_min)
        inc_maxs.append(inc_max)

    table = pa.table({
        "key.laser_name": pa.array(laser_names, type=pa.int32()),
        "key.segment_context_name": pa.array([SEGMENT] * 5),
        "[LiDARCalibrationComponent].extrinsic.transform":
            pa.array(extrinsics, type=pa.list_(pa.float64(), 16)),
        "[LiDARCalibrationComponent].beam_inclination.values":
            pa.array(inc_values, type=pa.list_(pa.float64())),
        "[LiDARCalibrationComponent].beam_inclination.min":
            pa.array(inc_mins, type=pa.float64()),
        "[LiDARCalibrationComponent].beam_inclination.max":
            pa.array(inc_maxs, type=pa.float64()),
    })
    pq.write_table(table, os.path.join(OUT_DIR, "lidar_calibration.parquet"))
    print(f"  lidar_calibration: 5 rows")


# ─── camera_calibration ─────────────────────────────────────────────────────

def gen_camera_calibration():
    cam_names = [1, 2, 3, 4, 5]
    heights = [1280, 1280, 1280, 886, 886]

    table = pa.table({
        "key.camera_name": pa.array(cam_names, type=pa.int32()),
        "key.segment_context_name": pa.array([SEGMENT] * 5),
        "[CameraCalibrationComponent].extrinsic.transform":
            pa.array([make_identity() for _ in range(5)], type=pa.list_(pa.float64(), 16)),
        "[CameraCalibrationComponent].width":
            pa.array([1920] * 5, type=pa.int32()),
        "[CameraCalibrationComponent].height":
            pa.array(heights, type=pa.int32()),
        "[CameraCalibrationComponent].intrinsic.f_u":
            pa.array([2000.0] * 5, type=pa.float64()),
        "[CameraCalibrationComponent].intrinsic.f_v":
            pa.array([2000.0] * 5, type=pa.float64()),
    })
    pq.write_table(table, os.path.join(OUT_DIR, "camera_calibration.parquet"))
    print(f"  camera_calibration: 5 rows")


# ─── lidar_box ──────────────────────────────────────────────────────────────

def gen_lidar_box():
    seg_names = []
    frame_ts = []
    obj_ids = []
    types = []
    cx, cy, cz = [], [], []
    sx, sy, sz = [], [], []
    headings = []

    type_choices = [1, 1, 1, 2, 2, 4]  # weighted: more vehicles

    for fi in range(NUM_FRAMES):
        for oi in range(OBJECTS_PER_FRAME):
            seg_names.append(SEGMENT)
            frame_ts.append(timestamps[fi])
            obj_ids.append(f"obj_{fi:03d}_{oi:03d}")
            types.append(int(np.random.choice(type_choices)))
            cx.append(float(np.random.uniform(-20, 50)))
            cy.append(float(np.random.uniform(-20, 20)))
            cz.append(float(np.random.uniform(0, 2)))
            sx.append(float(np.random.uniform(1, 5)))
            sy.append(float(np.random.uniform(1, 3)))
            sz.append(float(np.random.uniform(1, 3)))
            headings.append(float(np.random.uniform(-np.pi, np.pi)))

    table = pa.table({
        "key.segment_context_name": pa.array(seg_names),
        "key.frame_timestamp_micros": pa.array(frame_ts, type=pa.int64()),
        "key.laser_object_id": pa.array(obj_ids),
        "[LiDARBoxComponent].type": pa.array(types, type=pa.int32()),
        "[LiDARBoxComponent].box.center.x": pa.array(cx, type=pa.float64()),
        "[LiDARBoxComponent].box.center.y": pa.array(cy, type=pa.float64()),
        "[LiDARBoxComponent].box.center.z": pa.array(cz, type=pa.float64()),
        "[LiDARBoxComponent].box.size.x": pa.array(sx, type=pa.float64()),
        "[LiDARBoxComponent].box.size.y": pa.array(sy, type=pa.float64()),
        "[LiDARBoxComponent].box.size.z": pa.array(sz, type=pa.float64()),
        "[LiDARBoxComponent].box.heading": pa.array(headings, type=pa.float64()),
    })
    n = len(seg_names)
    pq.write_table(table, os.path.join(OUT_DIR, "lidar_box.parquet"))
    print(f"  lidar_box: {n} rows")


# ─── lidar (heavy) ──────────────────────────────────────────────────────────

def make_range_image(height, width):
    """Generate a mock range image: [H, W, 4] → flat list.
    Channels: [range, intensity, elongation, nlz_mask].
    ~88% pixels valid (range > 0), rest invalid (range = 0).
    Range values: 5–70m, intensity: 0–200, elongation: 0–3.
    """
    n = height * width
    valid = np.random.random(n) < VALID_RATIO

    ranges = np.where(valid, np.random.uniform(5, 70, n), 0.0).astype(np.float32)
    intensity = np.where(valid, np.random.uniform(0, 200, n), 0.0).astype(np.float32)
    elongation = np.where(valid, np.random.uniform(0, 3, n), 0.0).astype(np.float32)
    nlz = np.zeros(n, dtype=np.float32)  # no-label-zone mask (unused in tests)

    # Interleave: [r0, i0, e0, n0, r1, i1, e1, n1, ...]
    values = np.empty(n * 4, dtype=np.float32)
    values[0::4] = ranges
    values[1::4] = intensity
    values[2::4] = elongation
    values[3::4] = nlz

    shape = [height, width, 4]
    return shape, values.tolist()


def gen_lidar():
    seg_names = []
    frame_ts = []
    laser_names = []
    shapes = []
    values = []

    sensors = sorted(SENSOR_DIMS.keys())

    for fi in range(NUM_FRAMES):
        for ln in sensors:
            h, w = SENSOR_DIMS[ln]
            shape, vals = make_range_image(h, w)
            seg_names.append(SEGMENT)
            frame_ts.append(timestamps[fi])
            laser_names.append(ln)
            shapes.append(shape)
            values.append(vals)

    table = pa.table({
        "key.segment_context_name": pa.array(seg_names),
        "key.frame_timestamp_micros": pa.array(frame_ts, type=pa.int64()),
        "key.laser_name": pa.array(laser_names, type=pa.int32()),
        "[LiDARComponent].range_image_return1.shape":
            pa.array(shapes, type=pa.list_(pa.int32(), 3)),
        "[LiDARComponent].range_image_return1.values":
            pa.array(values, type=pa.list_(pa.float32())),
    })
    n = len(seg_names)
    # Write with row_group_size=200 to create ~5 row groups
    pq.write_table(
        table,
        os.path.join(OUT_DIR, "lidar.parquet"),
        row_group_size=200,
        compression="zstd",
    )
    print(f"  lidar: {n} rows (5 row groups, ZSTD)")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating fixtures in {OUT_DIR}")
    gen_vehicle_pose()
    gen_lidar_calibration()
    gen_camera_calibration()
    gen_lidar_box()
    gen_lidar()

    # Print sizes
    total = 0
    for f in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, f)
        size = os.path.getsize(path)
        total += size
        print(f"  {f}: {size / 1024:.1f} KB")
    print(f"  TOTAL: {total / 1024:.1f} KB")
