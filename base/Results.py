class BaseResults:
    def __init__(self):
        self.ids = None
        self.ages = None
        self.labels = None
        self.last_det_data = None
        self.last_det_points = None
        self.kwds:dict = None