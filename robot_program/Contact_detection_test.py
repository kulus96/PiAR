from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEInterface
from rtde_io import RTDEIOInterface as RTDEIO
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
from enum import IntEnum
import matplotlib.pyplot as plt

test_array1 = []
test_array2 = []

class axis(IntEnum):
    x = 0
    y = 1
    z = 2


class Robot:

    def __init__(self, ip, sampleFreq):
        self.control = RTDEControl(ip)
        self.interface = RTDEInterface(ip, [])
        self.io = RTDEIO(ip)
        self.sampleFreq = sampleFreq
        self.sampleTime = 1/self.sampleFreq

        # Collision detection
        self.contactThres = 4
        self.prevForce = 0


        # Sensor data
        self.FTMeasurements = []
        self.activeFMeasurement = [0] #[0]
        self.TCPPoses = []
        self.axes = np.array([[0], [1], [2]]) #x,y,z

    def collisionDetection(self, _axis = axis.z):

        currentForce = self.interface.getActualTCPForce()[2]
        print(currentForce,self.prevForce)
        test_array1.append(currentForce)
        df = currentForce - self.prevForce
        self.prevForce = currentForce
        if abs(df) > self.contactThres:
            test_array2.append(1)
            print(df)
            return True
        else:
            test_array2.append(0)
            return False


def runCollisionDetection(robot,startPos, collisionVel, collisionAcc, sampleTime):
    robot.control.moveL(startPos, collisionVel, collisionAcc, True)
    print(startPos)
    start_time = time.time()
    stop_loop = 0
    robot.prevForce = 0
    while not robot.collisionDetection() and not stop_loop:
        if time.time()-start_time > 5.0:
            stop_loop = 1
        time.sleep(sampleTime)

    if stop_loop :
        print("Collision was NOT detected")
        exit(-1)
    else:
        print("Collision was detected")
    robot.control.stopL(10.0)

if __name__ == '__main__':
    ip = "192.168.1.130" # ip for main

    sampleFreq = 500  # [Hz]
    robot = Robot(ip, sampleFreq)
    sampleTime = 1/sampleFreq

    desiredOrientation = R.from_euler('xyz', [np.pi-np.pi/4, 0, 0]).as_rotvec()

    # Collision detection
    collisionVel = 0.05  # [m/s]
    collisionAcc = 0.5  # [m/s^2]

    # Resetting sensors
    robot.control.zeroFtSensor()
    time.sleep(2)

    print(robot.interface.getActualTCPPose())
    print(robot.interface.getActualTCPForce())

    homePos = [0.5, -0.2, 0.13, -2.6530589595942846, 1.6348842894331148, 0.04280016391235699]
    homePos[3:] = desiredOrientation

    startPos = [0.5, -0.2, 0.13, -2.6530589595942846, 1.6348842894331148, 0.04280016391235699]
    startPos[2] -= 0.1  # [m] offset
    startPos[3:] = desiredOrientation

    numberOfRuns = 10
    

    robot.control.moveL(homePos)

    
    for i in range(numberOfRuns):
        
        robot.control.moveL(homePos)
        time.sleep(1)
        input("ready")

        
        # Starting collision detection
        print("Run collision detection")
        runCollisionDetection(robot, startPos, collisionVel, collisionAcc, sampleTime)
        time.sleep(0.5)
        robot.control.moveL(homePos)

    
    plt.plot(range(len(test_array1)),test_array1)
    plt.plot(range(len(test_array2)),test_array2)
    plt.show()
    
    np.save("Force data_z_10_45degree.npy",np.array(test_array1))
    np.save("collision_data_10_45degree.npy",np.array(test_array2))
    robot.control.servoStop()
    robot.control.speedStop()
    print("Ended program!")
