from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEInterface
from rtde_io import RTDEIOInterface as RTDEIO
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import time

class Robot(RTDEControl, RTDEInterface, RTDEIO):
    def __init__(self, ip):
        self.control = RTDEControl(ip)
        self.interface = RTDEInterface(ip)
        self.io = RTDEIO(ip)



if __name__ =='__main__':
    print("hello World!")
    Fx = np.array([])
    Fy = np.array([])
    Fz = np.array([])
    Tx = np.array([])
    Ty = np.array([])
    Tz = np.array([])
    ip = "192.168.1.111"
    robot = Robot(ip)
    # init_q = robot.interface.getActualQ()
    # print("inital q: ", init_q)
    dt = 1.0/500.0

    for t in range(5000): # 10 seconds
        start = time.time()
        robot.control.zeroFtSensor()
        TCP_pose = robot.interface.getActualTCPPose()
        rotm = R.from_rotvec(TCP_pose[3:]).as_matrix()
        TCP_force = np.asarray(robot.interface.getActualTCPForce())

        rotated_force = np.matmul(rotm,TCP_force[:3])
        rotated_torque = np.matmul(rotm,TCP_force[3:])
        Fx = np.append(Fx, rotated_force[0])
        Fy = np.append(Fy, rotated_force[1])
        Fz = np.append(Fz, rotated_force[2])
        Tx = np.append(Tx, rotated_torque[0])
        Ty = np.append(Ty, rotated_torque[1])
        Tz = np.append(Tz, rotated_torque[2])
        #print("data: " ,[rotated_force, rotated_torque])
        end = time.time()
        dur= end-start
        if  dur< dt:
            time.sleep(dt-dur)
    dict = {'Fx': Fx, 'Fy':Fy, 'Fz':Fz, 'Tx':Tx, 'Ty':Ty, 'Tz':Tz}
    df = pd.DataFrame(dict)
    df.to_csv('sensorNoise.csv', index=False)
