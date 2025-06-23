from ...base.Drawer import BaseDrawer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .results import norfairResults

from norfair.drawing.drawer import Drawer
from norfair.drawing.color import Palette
import numpy as np

class norfairDrawer(BaseDrawer):
    def __init__(self, results = None):
        super().__init__(results)

    def draw_info(self,
                  frame,
                  tracker_results:"norfairResults" = None,
                  draw_roi=True,
                  draw_id=True,
                  draw_points=True,
                  draw_bounding_box=True,
                  draw_estimate=True):
        draw_info = frame.copy()
        
        if tracker_results: # update
            self.Results = tracker_results

        if draw_roi:
            draw_info = self._draw_roi(draw_info, self.Results.roi)

        if self.Results.ids:
            for i, indx in enumerate(self.Results.ids):
                if draw_estimate:
                    estimate = self.Results.estimate[i]
                    draw_info = self._draw_estimate(frame=draw_info, 
                                                    point=estimate,
                                                    idx=indx, 
                                                    distance_threshold=self.Results.DISTANCE_THRESHOLD)

                point = self.Results.last_det_points[i]
                if draw_points:
                    draw_info = self._draw_point(draw_info, point, indx)
                if draw_id:
                    draw_info = self._draw_id(draw_info, indx, point)
                if draw_bounding_box:
                    box = self.Results.last_det_bounding_boxes[i]
                    draw_info = self._draw_box(draw_info, box, indx)
            
        return draw_info
    
    def _draw_estimate(self, 
                       frame, 
                       point, 
                       idx, 
                       distance_threshold,
                       radius=None, 
                       thickness=None):
        
        draw_estimate = frame.copy()
        color = Palette.choose_color(idx)
        point = np.array(point, dtype=np.int32).reshape(-1)

        if thickness is None:
            thickness = self._thickness_cal(frame)

        if radius is None:
            frame_scale = frame.shape[0] / 100
            radius = int(frame_scale * 0.3)

        if len(point) == 2:
            draw_estimate = Drawer.circle(draw_estimate, point, int(distance_threshold), thickness, color)
            draw_estimate = Drawer.circle(draw_estimate, point, 2, thickness, color)
        else:
            draw_estimate = Drawer.rectangle(draw_estimate, point.reshape(2,2),color, thickness)
            # center call 
            p1, p2 = point.reshape(2,2)
            ct = ((p2 - p1) // 2) + p1
            draw_estimate = Drawer.circle(draw_estimate, ct, int(distance_threshold), thickness, color)
        return super()._draw_point(draw_estimate, point, idx, radius, thickness)