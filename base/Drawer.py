import numpy as np
import cv2
from norfair.drawing.drawer import Drawer
from norfair.drawing.color import Palette
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Results import BaseResultsTracker

class BaseDrawer:
    def __init__(self, results:"BaseResultsTracker"=None):
        self.color_mapping_keys = {
            'roi':(0,255,0),
            'roni':(0,0,255)
        }
        self.Results:"BaseResultsTracker" = results

    @staticmethod
    def _thickness_cal(frame):
        return int(max(frame.shape) / 500)
    
    @staticmethod
    def _font_scale_cal(frame):
        return max(max(frame.shape) / 2000, 0.5)
    
    def _get_color_roi(self, key):
        if key in self.color_mapping_keys:
            return self.color_mapping_keys[key]
        for k in self.color_mapping_keys:
            if k in key:
                return self.color_mapping_keys[k]
        return None

    def draw_tracker_results(self,
                  frame,
                  tracker_results:"BaseResultsTracker" = None,
                  draw_roi=True,
                  draw_id=True,
                  draw_points=True,
                  draw_bounding_box=True,):
        im = frame.copy()
        
        if tracker_results: # update
            self.Results = tracker_results

        if draw_roi:
            im = self._draw_roi(im, self.Results.roi)

        if self.Results.ids:
            for i, indx in enumerate(self.Results.ids):
                point = self.Results.last_det_points[i]
                if draw_points:
                    im = self._draw_point(im, point, indx)
                if draw_bounding_box:
                    box = self.Results.last_det_bounding_boxes[i]
                    im = self._draw_box(im, box, indx)
                if draw_id:
                    im = self._draw_id(im, indx, point)
        return im

    def _draw_id(self, frame, idx, position, size=None, thickness=None):
        if thickness is None:
            thickness = self._thickness_cal(frame)

        if size is None:
            size = max(max(frame.shape) / 2000, 0.5)

        draw_id = frame.copy()
        color = Palette.choose_color(idx)
        position = np.array(position, dtype=np.int32).reshape(-1)
        if len(position) == 4: # get center
            p1, p2 = position.reshape(2,2)
            position = ((p2 - p1) // 2) + p1
        draw_id = Drawer.text(draw_id, f'{idx}', position, size=size, color=color, thickness=thickness)
        return draw_id
    
    def _draw_box(self, frame, box, idx, thickness=None):
        if thickness is None:
            thickness = self._thickness_cal(frame)
        draw_box = frame.copy()
        box = np.array(box, dtype=np.int32).reshape(2,2)
        color = Palette.choose_color(idx)
        draw_box = Drawer.rectangle(draw_box, box, color, thickness)
        return draw_box

    def _draw_point(self, frame, point, idx, radius=None, thickness=None):
        if thickness is None:
            thickness = self._thickness_cal(frame)

        if radius is None:
            frame_scale = frame.shape[0] / 100
            radius = int(frame_scale * 0.3)
        
        draw_point = frame.copy()
        point = np.array(point, dtype=np.int32).reshape(-1)

        color = Palette.choose_color(idx)

        if len(point) == 2:
            draw_point = Drawer.circle(draw_point, point, radius, thickness, color)
        elif len(point) == 4:
            draw_point = Drawer.rectangle(draw_point, point.reshape(2,2), color, thickness)
        return draw_point
        
    def _draw_roi(self,
                 frame,
                 roi_dict,
                 font_size=1, 
                 font_thickness=2,
                 font_color=(255,255,255)):
        draw_roi = frame.copy()
        overlay = frame.copy()
        h, w = draw_roi.shape[:2]

        if roi_dict:
            for key, val in roi_dict.items():
                color = self._get_color_roi(key)
                if color is None:
                    continue

                alpha_fill = 0.3
                np_val = np.array(val)
                abs_val = np_val * [w, h]
                abs_val = abs_val.astype(np.int32)
                cv2.fillPoly(overlay, [abs_val], color=color)
                cv2.polylines(draw_roi, [abs_val], isClosed=True, color=color, thickness=2)

                M = cv2.moments(abs_val)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = abs_val[0]

                Drawer.text(draw_roi, key, (cX, cY), font_size, font_color, font_thickness)

            cv2.addWeighted(overlay, alpha_fill, draw_roi, 1 - alpha_fill, 0, draw_roi)
        return draw_roi