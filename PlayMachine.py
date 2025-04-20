#machines should be an array of polygons representing the machines, body_cm is the coordinates of the center of mass of the person
from shapely.geometry import Polygon
from shapely.geometry import Point
from Bandits import Casino, Bandit

polygon_points = [
    [(0.98, 0.48), (0.95, 0.19), (0.65, 0.2), (0.68, 0.5)], #top right, top left, bottom left, bottom right
    [(0.95, 0.19), (0.96, -0.16), (0.63, -0.15),(0.65, 0.2)], 
    [(0.96, -0.16), (0.96, -0.41), (0.61, -0.4), (0.63, -0.15)],
    [(0.96, -0.41), (0.9, -0.71), (0.6, -0.7), (0.61, -0.4)]
]

# Create Shapely polygons
machines = [Polygon(points) for points in polygon_points]

def PlayMachine(machines, body_cm, casino):
	machine_played = 0
	won = None
	for i,machine in enumerate(machines):
		if machine.contains(body_cm):
			machine_played = i+1
	if machine_played != 0:
		#the -1 is bc python does zero indexing
		won = casino.bandits[machine_played-1].pull()
	return machine_played, won
