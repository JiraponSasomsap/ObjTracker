from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict

@dataclass
class BaseResults:
    ids: Optional[List[Any]] = None
    ages: Optional[List[int]] = None
    labels: Optional[List[Any]] = None
    last_det_data: Optional[List[dict]] = None
    last_det_points: Optional[List[Any]] = None
    last_det_bounding_boxes: Optional[List[Any]] = None
    roi: Optional[Dict] = None