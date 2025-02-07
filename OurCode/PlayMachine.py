#machines should be an array of polygons representing the machines, body_cm is the coordinates of the center of mass of the person
from shapely.geometry import Polygon
from shapely.geometry import Point
machine1c= [];
machine2c=[];
machine3c=[];
machine4c=[];
body_c=[];
machine1=Polygon(machine1c)
machine2=Polygon(machine2c)
machine3=Polygon(machine3c)
machine4=Polygon(machine4c)
body_cm=Point(body_c)
machines=[machine1,machine2,machine3,machine4]
def PlayMachine(machines, body_cm):
for i,machine in enumerate(machines):
	if machine.contains(body_cm):
		return i+1
return 0
