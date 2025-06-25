from norfair.tracker import (TrackedObject, 
                             Tracker, 
                             _TrackedObjectFactory, 
                             Detection)
from norfair.filter import OptimizedKalmanFilterFactory
import numpy as np
import sys
import os
import warnings
from shapely.geometry import Polygon, Point
from pathlib import Path
import yaml
from datetime import datetime

try:
    from ...base.Results import BaseResults
    from ...base.Drawer import BaseDrawer
except:
    warnings.warn("Relative imports failed. Falling back to absolute imports.")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
    from base.Results import BaseResults
    from base.Drawer import BaseDrawer

from .drawer import norfairDrawer
from .results import norfairResults
from dataclasses import dataclass

class norfairDevTrackedObject(TrackedObject):
    def __init__(self, 
                 obj_factory, 
                 initial_detection, 
                 hit_counter_max, 
                 initialization_delay, 
                 pointwise_hit_counter_max, 
                 detection_threshold, 
                 period, 
                 filter_factory, 
                 past_detections_length, 
                 reid_hit_counter_max, 
                 coord_transformations = None):
        super().__init__(obj_factory, 
                         initial_detection, 
                         hit_counter_max, 
                         initialization_delay, 
                         pointwise_hit_counter_max, 
                         detection_threshold, 
                         period, 
                         filter_factory, 
                         past_detections_length, 
                         reid_hit_counter_max, 
                         coord_transformations)

class norfairDevTracker(Tracker):
    def __init__(self,
                 distance_function, 
                 distance_threshold, 
                 hit_counter_max = 15, 
                 initialization_delay = None, 
                 pointwise_hit_counter_max = 4, 
                 detection_threshold = 0, 
                 filter_factory:'OptimizedKalmanFilterFactory' = None, 
                 past_detections_length = 4, 
                 reid_distance_function = None, 
                 reid_distance_threshold = 0, 
                 reid_hit_counter_max = None):
        
        if filter_factory is None:
            filter_factory = OptimizedKalmanFilterFactory()

        self.Results = norfairResults()
        self.Drawer = norfairDrawer(self.Results)

        self._config = {
            'distance_function':distance_function if isinstance(distance_function, str) else distance_function.__class__.__name__,
            'distance_threshold':distance_threshold,
            'hit_counter_max':hit_counter_max,
            'initialization_delay':initialization_delay,
            'pointwise_hit_counter_max':pointwise_hit_counter_max,
            'detection_threshold':detection_threshold,
            'filter_factory':filter_factory.__class__.__name__,
            'past_detections_length':past_detections_length,
            'reid_distance_function':reid_distance_function.__class__.__name__ if reid_distance_function is not None else None,
            'reid_distance_threshold':reid_distance_threshold,
            'reid_hit_counter_max':reid_hit_counter_max,
        }

        self.callback = Callback()

        super().__init__(distance_function, 
                         distance_threshold, 
                         hit_counter_max, 
                         initialization_delay, 
                         pointwise_hit_counter_max, 
                         detection_threshold, 
                         filter_factory, 
                         past_detections_length, 
                         reid_distance_function, 
                         reid_distance_threshold, 
                         reid_hit_counter_max)

    def set_tracker(self, 
                    custom_tracked_object = norfairDevTrackedObject, 
                    color_mapping_keys={
                        'roi':(0,255,0),
                        'roni':(0,0,255)
                    },
                    **roi):
        '''
        Set custom tracked object factory and region configs.

        Parameters:
            custom_tracked_object: class to use for tracked objects
            **roi: additional region configurations (e.g., roi, roni, roi1, roni1)

        Notes:
            - "roi" = Region Of Interest
            - "roni" = Region Of No Interest
        '''
        setattr(self, '_obj_factory', _norfairDevTrackedObjectAutoFactory(custom_tracked_object))
        self.Results.roi = roi 
        self.Results.DISTANCE_THRESHOLD = self.distance_threshold
        self.Drawer.color_mapping_keys = color_mapping_keys
        return self
    
    def save_config(self, dst='.'):
        dt = datetime.now().strftime("%d-%m-%Y-%H%M%S")
        dst = Path(dst) / f'tracker_config/{self.__class__.__name__} {dt}.yaml'
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        return self
    
    def _preprocess_update_input(self, frame, points, scores, data, label, embedding):
        if self.Results.roi is None:
            return points, scores, data, label, embedding

        if all(not 'roi' in key for key in self.Results.roi):
            return points, scores, data, label, embedding
        
        h, w = frame.shape[:2]

        new_points = []
        new_scores = [] if scores is not None else None
        new_data = [] if data is not None else None
        new_label = [] if label is not None else None
        new_embedding = [] if embedding is not None else None

        for i in range(len(points)):
            point = points[i]
            center = None
            if len(point) == 4:
                p1, p2 = np.array(point).reshape(2, 2)
                center = Point(((p2 - p1) / 2) + p1)
            elif len(point) == 2:
                center = Point(np.array(point))
            else:
                raise ValueError(f"Unexpected point format at index {i}: {point}")
            
            for key, val in self.Results.roi.items():
                if 'roi' in key:
                    arr = np.array(val) * [w, h]
                    arr = np.vstack([arr, arr[0]])
                    roi = Polygon(arr)
                    if center.within(roi):
                        new_points.append(points[i])
                        if new_scores is not None:
                            new_scores.append(scores[i])
                        if new_label is not None:
                            new_label.append(label[i])
                        if new_embedding is not None:
                            new_embedding.append(embedding[i])
                        if new_data is not None:
                            new_data.append(data[i])
                        break
        return (np.array(new_points),
                new_scores,
                new_data,
                new_label,
                new_embedding,)

    def update_detections(
        self,
        frame:np.ndarray,
        points:np.ndarray|list,
        scores:np.ndarray|list=None,
        data:np.ndarray|list=None,
        label:np.ndarray|list=None,
        embedding:np.ndarray|list=None,
        **update_params
    ):
        '''
        Convenient wrapper to convert raw detection data into Detection objects 
        and update the tracker.

        Parameters:
            points (np.ndarray): An (N, 2) or (N, 4) array of detection points.
                - (x, y) for keypoints or center points
                - (x1, y1, x2, y2) for bounding boxes
            scores (list or array, optional): Confidence scores per detection.
            data (list, optional): Additional data for each detection.
            label (list, optional): Class labels per detection.
            embedding (list, optional): Feature vectors for Re-ID (Re-identification).
            **update_params: Extra parameters passed to the underlying `update()`.

        Returns:
            list: List of updated `TrackedObject` instances.
        
        Raises:
            TypeError: If `points` is not convertible to a numpy array.
            ValueError: If shape of `points` is invalid or if any extra param
                        (scores, data, label, embedding) has mismatched length.
        '''
        if not isinstance(points, np.ndarray):
            try:
                points = np.array(points)
            except Exception as e:
                raise TypeError("points must be convertible to a numpy.ndarray") from e

        # if points.ndim != 2 or points.shape[1] not in (2, 4):
        #     raise ValueError("points must be a 2D array with shape (N, 2) or (N, 4)")

        param_dict = {
            'scores': scores,
            'data': data,
            'label': label,
            'embedding': embedding
        }

        for name, param in param_dict.items():
            if param is not None and len(param) != len(points):
                raise ValueError(f"Length mismatch: {name} has length {len(param)} but points has length {len(points)}")
            
        (
            points, scores, data, label, embedding
        ) = self._preprocess_update_input(
            frame=frame,
            points=points, 
            scores=scores, 
            data=data, 
            label=label, 
            embedding=embedding, 
        )

        detections = []
        for i in range(len(points)):
            point = points[i]
            if len(point) == 4:
                point = point.reshape(2, 2)
            detections.append(
                Detection(
                    points=point,
                    scores=scores[i] if scores is not None else None,
                    data=data[i] if data is not None else None,
                    label=label[i] if label is not None else None,
                    embedding=embedding[i] if embedding is not None else None
                )
            )
        self.update(detections=detections, bounding_boxes_input=points, **update_params)
        return self.Results
    
    def update(self, detections = None, bounding_boxes_input = None, period = 1, coord_transformations = None):
        self.Results.bounding_boxes_input = bounding_boxes_input # subscribe bounding boxes input
        return super().update(detections, period, coord_transformations)
    
    def _update_tracker_results(self):
        result_dict = {
            'ids': [],
            'ages': [],
            'labels': [],
            'last_det_data': [],
            'last_det_points': [],
            'last_det_bounding_boxes': [],
            'estimate': [],
            'hit_counter': [],
            'is_update_detections': [],
        }

        for obj in self.get_active_objects():
            result_dict['ids'].append(obj.id)
            result_dict['ages'].append(obj.age)
            result_dict['labels'].append(obj.label)
            result_dict['last_det_data'].append(obj.last_detection.data)
            result_dict['last_det_points'].append(obj.last_detection.points)
            result_dict['last_det_bounding_boxes'].append(
                obj.last_detection.data.get('boxes')
            )
            result_dict['estimate'].append(obj.estimate)

            if obj.id in self.Results.ids:
                index = self.Results.ids.index(obj.id)
                hit = self.Results.hit_counter[index]
                result_dict['is_update_detections'].append(obj.hit_counter >= hit)
            else:
                result_dict['is_update_detections'].append(True)

            result_dict['hit_counter'].append(obj.hit_counter)

            if callable(self.callback._update_tracker_results):
                self.callback._update_tracker_results(obj, result_dict)

        # assign values back to self.Results
        for k, v in result_dict.items():
            setattr(self.Results, k, v)

    def __getattribute__(self, name):
        if name == 'update':
            object.__getattribute__(self, '_update_tracker_results')()
        return object.__getattribute__(self, name)

class _norfairDevTrackedObjectAutoFactory(_TrackedObjectFactory):
    def __init__(self, object_class: type):
        super().__init__()
        if not issubclass(object_class, norfairDevTrackedObject):
            raise TypeError("object_class must be a subclass of TrackedObject")
        self.object_class = object_class

    def create(
        self,
        initial_detection: "Detection",
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory,
        past_detections_length: int,
        reid_hit_counter_max,
        coord_transformations,
    ) -> TrackedObject:
        return self.object_class(
            obj_factory=self,
            initial_detection=initial_detection,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=pointwise_hit_counter_max,
            detection_threshold=detection_threshold,
            period=period,
            filter_factory=filter_factory,
            past_detections_length=past_detections_length,
            reid_hit_counter_max=reid_hit_counter_max,
            coord_transformations=coord_transformations,
        )
    
@dataclass
class Callback:
    _update_tracker_results = None