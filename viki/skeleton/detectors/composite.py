# abstract method but didnt show explicitly, looks ugly
from abc import abstractmethod
from enum import Enum
from typing import Optional

from viki.skeleton.detectors.base import PartialDetection2D
from viki.skeleton.models import PreparedFrame

class FusionMode(str, Enum):
    ANY = "any"  # if any detector detects, return the detection
    ALL = "all"  # if all detectors detect, return the detection

class CompositeLandmarkDetector():
    def __init__(self, detectors: list[PartialLandmarkDetector], mode: FusionMode = FusionMode.ANY):
        self.detectors = detectors
        self.mode = mode
    
    def detect(self, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        detections: list[PartialDetection2D] = []
        for detector in self.detectors:
            detection = detector.detect(frame)

            if detection is not None:
                detections.append(detection)
                
        
        if self.mode == FusionMode.ANY:
            if detections:
                return detections #return all found detections
        elif self.mode == FusionMode.ALL:
            if len(detections) == len(self.detectors):
                return detections #return all found detections
        
        return None
            
    def close(self) -> None:
        for detector in self.detectors:
            detector.close()


class PartialLandmarkDetector():
    name: str
    indices: tuple[int, ...]
    priority: int # to solve conflicts                

    @abstractmethod
    def detect(self, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        pass

    def close(self) -> None:
        pass
    