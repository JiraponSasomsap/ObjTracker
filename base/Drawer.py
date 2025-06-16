import numpy as np
import cv2
from norfair.drawing.drawer import Drawer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Results import BaseResults

class BaseDrawer:
    def __init__(self, results:"BaseResults"=None):
        self.color_mapping_keys = {
            'roi':(0,255,0),
            'roni':(0,0,255)
        }
        self.Results:"BaseResults" = results

    def get_color(self, key):
        if key in self.color_mapping_keys:
            return self.color_mapping_keys[key]
        for k in self.color_mapping_keys:
            if k in key:
                return self.color_mapping_keys[k]
        return (0,0,0)

    def draw_roi(self,
                 img,
                 tracker_results:"BaseResults" = None,
                 font_size=1, 
                 font_thickness=2,
                 font_color=(255,255,255)):
        im = img.copy()
        overlay = img.copy()
        h, w = im.shape[:2]

        kwds = None
        if tracker_results or self.Results:
            if tracker_results is not None:
                kwds = tracker_results.Results.kwds
            else:
                kwds = self.Results.kwds

        if kwds:
            for key, val in kwds.items():
                color = self.get_color(key)
                if color is None:
                    continue

                alpha_fill = 0.3
                np_val = np.array(val)
                abs_val = np_val * [w, h]
                abs_val = abs_val.astype(np.int32)
                cv2.fillPoly(overlay, [abs_val], color=color)
                cv2.polylines(im, [abs_val], isClosed=True, color=color, thickness=2)

                M = cv2.moments(abs_val)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = abs_val[0]

                Drawer.text(im, key, (cX, cY), font_size, font_color, font_thickness)

            cv2.addWeighted(overlay, alpha_fill, im, 1 - alpha_fill, 0, im)
        return im