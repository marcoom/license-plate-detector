# src/tracking/tracker.py
from src.third_party.deep_sort.deep_sort.tracker import Tracker

class DeepSortTracker:
    def __init__(self):
        self.deepsort = Tracker(max_age=30)

    def update(self, detections, frame):
        tracks = self.deepsort.update_tracks(detections, frame=frame)
        return tracks
