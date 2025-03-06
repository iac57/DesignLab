#machines should be an array of polygons representing the machines, body_cm is the coordinates of the center of mass of the person
from shapely.geometry import Polygon
from shapely.geometry import Point
from Bandits import Casino, Bandit

polygon_points = [
    [(-2.51, 0.4), (-2.51, 1), (-3.1, 1), (-3.1, 0.4)],
    [(-2.51, 1), (-2.51, 1.61), (-3.1, 1.61), (-3.1, 1)],
    [(-2.51, 1.61), (-2.51, 2.2), (-3.1, 2.2), (-3.1, 1.61)],
    [(-2.51, 2.2), (-2.51, 2.8), (-3.1, 2.8), (-3.1, 2.2)]
]

# Create Shapely polygons
machines = [Polygon(points) for points in polygon_points]

M = .2
B = 1
bandit1 = Bandit(B/4)
bandit2 = Bandit(B/4)
bandit3 = Bandit(B/4)
bandit4 = Bandit(B/4)
casino = Casino([bandit1, bandit2, bandit3, bandit4], B, M)

def PlayMachine(machines, body_cm):
	machine_played = 0
	won = None
	for i,machine in enumerate(machines):
		if machine.contains(body_cm):
			machine_played = i+1
	if machine_played != 0:
		casino.setPayoutsRandom()
		#the -1 is bc python does zero indexing
		won = casino.bandits[machine_played-1].pull()
	return machine_played, won
