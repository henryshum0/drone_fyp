import numpy as np

class TrackSettings():
    def __init__(self, waypoints_xyz, waypoints_rpy, easy_settings, medium_settings, hard_settings):
        self.waypoints_xyz = waypoints_xyz
        self.waypoints_rpy = waypoints_rpy
        self.easy_settings = easy_settings
        self.medium_settings = medium_settings
        self.hard_settings = hard_settings