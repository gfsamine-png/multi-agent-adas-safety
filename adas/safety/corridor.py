from __future__ import annotations

# Part of the frame width (left, right) where our lane must be, used only
# when the lane lines cannot place an object.
DEFAULT_CORRIDOR = (0.30, 0.70)


def effective_lane_position(
    lane_pos: str,
    bbox: tuple[float, float, float, float],
    frame_width: int | None,
    corridor: tuple[float, float] = DEFAULT_CORRIDOR,
) -> str:
    """Fallback for objects whose lane position is 'unknown'.

    An object centered far to the left or right of the image cannot be in our
    lane, so it is classed as outside. Known positions are never changed.
    """
    if lane_pos != "unknown" or not frame_width:
        return lane_pos
    cx = (bbox[0] + bbox[2]) / 2.0
    left, right = corridor
    if cx < left * frame_width:
        return "left_of_lane"
    if cx > right * frame_width:
        return "right_of_lane"
    return "unknown"