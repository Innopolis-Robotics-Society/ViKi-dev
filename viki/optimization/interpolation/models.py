from pydantic import BaseModel


class RecordedSkeletonFrame(BaseModel):
    ts: int
    landmarks: dict[int, list[float | None]]
