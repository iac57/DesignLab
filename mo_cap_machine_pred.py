import numpy as np
from scipy.spatial.transform import Rotation as R
#thetas should be an array of 5 angles from 0 to 180 degrees
thetas = []
#def classify_torso_angle(quaternion, thetas):
#    euler_angles = R.from_quat(quaternion).as_euler('zyx', degrees=True)
#    torso_angle = euler_angles[1]
#
#    for i in range(1, len(thetas)):
#        if thetas[i-1] <= torso_angle < thetas[i]:
#            return i  
#    print("Is the subject facing the wrong way??")
#    return None 
def classify_torso_angle(quaternion, thetas):
    euler_angles = R.from_quat(quaternion).as_euler('zyx', degrees=True)
    torso_angle = euler_angles[1]
    if torso_angle < 0:
        answer = 12
    else:
        answer = 34
    return answer