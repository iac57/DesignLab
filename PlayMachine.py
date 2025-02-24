#machines should be an array of polygons representing the machines, body_cm is the coordinates of the center of mass of the person
from shapely.geometry import Polygon
from shapely.geometry import Point

polygon_points = [
    [(-2.51, 0.4), (-2.51, 1), (-3.1, 1), (-3.1, 0.4)],
    [(-2.51, 1), (-2.51, 1.61), (-3.1, 1.61), (-3.1, 1)],
    [(-2.51, 1.61), (-2.51, 2.2), (-3.1, 2.2), (-3.1, 1.61)],
    [(-2.51, 2.2), (-2.51, 2.8), (-3.1, 2.8), (-3.1, 2.2)]
]

# Create Shapely polygons
machines = [Polygon(points) for points in polygon_points]

def PlayMachine(machines, body_cm):
	for i,machine in enumerate(machines):
		if machine.contains(body_cm):
			return i+1
	return 0
