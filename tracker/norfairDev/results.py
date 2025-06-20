from ...base.Results import BaseResults
from typing import List, Any, Optional
from dataclasses import dataclass

@dataclass
class norfairResults(BaseResults):
    estimate: Optional[List[Any]] = None
    hit_counter: List[int] = None
    DISTANCE_THRESHOLD:float = None