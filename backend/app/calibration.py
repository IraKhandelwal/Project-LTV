"""Experimental planar ground mapping for operator-drawn camera references.

This computes ground area for drawn image polygons. It does not estimate people
or validate that the reference points really describe a flat rectangle.
"""

from __future__ import annotations

from typing import Any


def _solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small dense system with pivoted Gaussian elimination."""
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            raise ValueError("Reference corners do not define a usable rectangle")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [a - factor * b for a, b in zip(
                augmented[row], augmented[column], strict=True
            )]
    return [augmented[row][-1] for row in range(size)]


def _homography(corners: list[dict[str, float]], width: float, depth: float) -> list[float]:
    # The operator clicks the four image corners in order around the edge.
    ground = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
    rows: list[list[float]] = []
    values: list[float] = []
    for point, (gx, gy) in zip(corners, ground, strict=True):
        x, y = float(point["x"]), float(point["y"])
        rows.append([x, y, 1, 0, 0, 0, -gx * x, -gx * y])
        values.append(gx)
        rows.append([0, 0, 0, x, y, 1, -gy * x, -gy * y])
        values.append(gy)
    return _solve(rows, values) + [1.0]


def _project(point: dict[str, float], transform: list[float]) -> tuple[float, float]:
    x, y = float(point["x"]), float(point["y"])
    denominator = transform[6] * x + transform[7] * y + transform[8]
    if abs(denominator) < 1e-10:
        raise ValueError("A point cannot be mapped through this reference")
    return (
        (transform[0] * x + transform[1] * y + transform[2]) / denominator,
        (transform[3] * x + transform[4] * y + transform[5]) / denominator,
    )


def _inside_convex_polygon(point: dict[str, float], polygon: list[dict[str, float]]) -> bool:
    signs = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        cross = ((following["x"] - current["x"]) * (point["y"] - current["y"])
                 - (following["y"] - current["y"]) * (point["x"] - current["x"]))
        if abs(cross) > 1e-8:
            signs.append(cross > 0)
    return not signs or all(sign == signs[0] for sign in signs)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2


def describe_calibration(setup: dict[str, Any]) -> dict[str, Any]:
    """Return estimated ground areas for zones inside a marked flat rectangle."""
    reference = setup.get("ground_reference")
    zones = setup.get("zones", [])
    if not reference:
        return {"status": "uncalibrated", "reference_area_m2": None, "zones": []}
    corners = reference.get("corners", [])
    width, depth = float(reference.get("width_m", 0)), float(reference.get("depth_m", 0))
    if len(corners) != 4 or width <= 0 or depth <= 0:
        return {"status": "invalid", "reference_area_m2": None, "zones": []}
    try:
        transform = _homography(corners, width, depth)
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid", "reference_area_m2": None, "zones": []}

    results = []
    for zone in zones:
        points = zone.get("points", [])
        if len(points) < 3 or not all(_inside_convex_polygon(point, corners) for point in points):
            results.append({"zone_id": zone.get("zone_id"), "name": zone.get("name"),
                            "area_m2": None, "status": "outside-reference"})
            continue
        try:
            projected = [_project(point, transform) for point in points]
            area = _polygon_area(projected)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            area = 0
        results.append({"zone_id": zone.get("zone_id"), "name": zone.get("name"),
                        "area_m2": round(area, 2) if area > 0 else None,
                        "status": "area-estimate" if area > 0 else "invalid-zone"})
    return {
        "status": "experimental-area-mapping",
        "reference_area_m2": round(width * depth, 2),
        "zones": results,
        "notice": "Estimated planar area only; this does not estimate people or validate camera calibration.",
    }
