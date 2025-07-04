from ...base.Drawer import BaseDrawer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .results import norfairResults

from norfair.drawing.drawer import Drawer
from norfair.drawing.color import Palette
import numpy as np
from dataclasses import dataclass

@dataclass
class DrawerCallback:
    draw_tracker_results = None

class norfairDrawer(BaseDrawer):
    def __init__(self, results = None):
        super().__init__(results)
        self.callback = DrawerCallback()

    def draw_tracker_results(self,
                  frame,
                  tracker_results:"norfairResults" = None,
                  draw_roi=True,
                  draw_id=True,
                  draw_points=True,
                  draw_bounding_box=True,
                  draw_estimate=True,
                  use_callback=True):
        im = frame.copy()
        
        if tracker_results: # update
            self.Results = tracker_results

        if draw_roi:
            im = self._draw_roi(im, self.Results.roi)

        if self.Results.ids:
            for i, indx in enumerate(self.Results.ids):
                if draw_estimate:
                    if not self.Results.is_update_detections[i]:
                        estimate = self.Results.estimate[i]
                        im = self._draw_estimate(frame=im, 
                                                        point=estimate,
                                                        idx=indx, 
                                                        distance_threshold=self.Results.DISTANCE_THRESHOLD)
                if self.Results.is_update_detections[i]:
                    point = self.Results.last_det_points[i]
                    if draw_points:
                        im = self._draw_point(im, point, indx)
                    if draw_id:
                        im = self._draw_id(im, indx, point)
                    if draw_bounding_box:
                        box = self.Results.last_det_bounding_boxes[i]
                        im = self._draw_box(im, box, indx)
                if callable(self.callback.draw_tracker_results) and use_callback:
                    im = self.callback.draw_tracker_results(im, self.Results, i)
        return im
    
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