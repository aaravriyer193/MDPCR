# MDPCR — core/__init__.py
from core.point import Point, make_point
from core.influence import InfluenceLayer, compute_influence_matrix
from core.convergence import ConvergenceTracker
from core.cloud import Cloud, cloud_distance, cloud_centroid
