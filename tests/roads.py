"""Test helpers for where things are on a track."""

import numpy as np

from mlracecar.core.geometry import FloatArray, cross, project_onto_polyline, unit_vector
from mlracecar.core.track.model import Track


def road_coordinates(track: Track, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Distance along the track and sideways offset (left = +) of each point, found exactly.

    Projecting onto the sampled centerline alone is off along the road by up to
    ``offset * curvature * spacing / 2``: about 0.15 m for a point 5 m off-center in a 12 m
    radius bend. Newton steps on the smooth spline then find the spot whose sideways line
    passes through the point.
    """
    arc_length = project_onto_polyline(points, track.centerline.points, closed=True).arc_length
    for _ in range(3):
        pose = track.pose_at(arc_length)
        tangent = unit_vector(pose.heading)
        gap = points - pose.position
        along, offset = np.einsum("ni,ni->n", gap, tangent), cross(tangent, gap)
        curvature = track.spline.curvature(*track.spline.locate(arc_length))
        arc_length = arc_length + along / (1 - curvature * offset)  # Newton step towards along = 0
    pose = track.pose_at(arc_length)
    return arc_length, cross(unit_vector(pose.heading), points - pose.position)
