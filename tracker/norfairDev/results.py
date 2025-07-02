from ...base.Results import BaseResultsTracker
from typing import List, Any, Optional
from dataclasses import dataclass

@dataclass
class norfairResults(BaseResultsTracker):
    estimate: Optional[List[Any]] = None
    hit_counter: List[int] = None
    DISTANCE_THRESHOLD:float = None
    is_update_detections: List[int] = None