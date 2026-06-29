from typing import TypeGuard
from pydantic import TypeAdapter

from viki.optimization.interpolation.models import RecordedSkeletonFrame


def is_float_list(vec: list[float | None]) -> TypeGuard[list[float]]:
    return not None in vec


class Interpolator:
    def __init__(self):
        pass

    def process(self, data: list[dict]) -> list[dict]:
        adapter = TypeAdapter(list[RecordedSkeletonFrame])
        frames = adapter.validate_python(data)
        prev_known_vecs: dict[int, list[float]] = {}
        prev_known_ts: dict[int, int] = {}
        pending: dict[int, list[RecordedSkeletonFrame]] = {}
        for frame in frames:
            for idx, vec in frame.landmarks.items():
                # vec contains None. can't interpolate if no known vec before
                if not is_float_list(vec):
                    if idx in prev_known_vecs:
                        pending.setdefault(idx, []).append(frame)
                    continue

                prev_vec = prev_known_vecs.get(idx)
                if not prev_vec:  # is None or empty (unlikely)
                    prev_known_vecs[idx] = vec
                    prev_known_ts[idx] = frame.ts
                    continue

                pending_frames = pending.get(idx)
                if not pending_frames:  # is None or empty
                    prev_known_vecs[idx] = vec
                    prev_known_ts[idx] = frame.ts
                    continue

                prev_ts = prev_known_ts.get(idx)
                if prev_ts is None:  # can be 0
                    prev_known_vecs[idx] = vec
                    prev_known_ts[idx] = frame.ts
                    continue

                cur_ts = frame.ts
                delta_ts = cur_ts - prev_ts
                weight_zero = False
                if delta_ts == 0.0:  # previous and next have same timestamp (unlikely)
                    weight_zero = True

                for interp_frame in pending_frames:
                    interp_vec = interp_frame.landmarks.get(idx)
                    if not interp_vec:  # is None or empty
                        continue

                    if weight_zero:
                        weight = 0
                    else:
                        weight = (interp_frame.ts - prev_ts) / delta_ts

                    for i, val in enumerate(interp_vec):  # interpolate
                        if val is None:
                            interp_vec[i] = prev_vec[i] + weight * (
                                vec[i] - prev_vec[i]
                            )
                    interp_frame.landmarks[idx] = interp_vec
                pending[idx] = []

                prev_known_vecs[idx] = vec
                prev_known_ts[idx] = frame.ts
        return adapter.dump_python(frames)
