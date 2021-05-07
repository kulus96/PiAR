from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEInterface
from rtde_io import RTDEIOInterface as RTDEIO
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter,lfilter_zi,filtfilt
import numpy as np
import pandas as pd
import time
from enum import IntEnum
import matplotlib.pyplot as plt
import math
from scipy import zeros, signal, random


def lp_filter(filter_input, filter_state, lpf_alpha):
    """Low-pass filter

        Args:
          filter_input ([]): input to be filtered
          filter_state ([]): initial filter state
          lpf_alpha (float): LPF alpha

        Returns:
          filter_state : the filter state
        """
    filter_out = filter_state - (lpf_alpha * (filter_state - filter_input))
    return filter_out

def calculate_lpf_alpha(cutoff_frequency, dt):
    """Low-pass filter

        Args:
          cutoff_frequency (float): cut
          dt (float): timestep dt

        Returns:
          LPF alpha (float)
        """
    return (2 * math.pi * dt * cutoff_frequency) / (2 * math.pi * dt * cutoff_frequency + 1)


test_array1 = []
test_array2 = []
test_array3 = []
test_array4 = []
test_array5 = []
test_array6 = []



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

        # Force controller
        self.kF = 10**(-5) #0.0002 #10**(-7) #10**(-5)     # [m/N]
        self.dF = 0#10**(-5)/20 #self.kF * 15
        self.iF = 0#1*10**(-7)*0.6 #self.kF * 0.6 #10**(-2)
        self.prevForce = 0
        self.prevIntegral = 0

        # Sensor data
        self.FTMeasurements = []
        self.activeFMeasurement = [] #[0]
        self.TCPPoses = []
        self.axes = np.array([[0], [1], [2]]) #x,y,z

        #Filter setup
        self.b = signal.firwin(numtaps = 50, cutoff= 1, fs = 500)
        self.z = signal.lfilter_zi(self.b, 1)


    def collectRawFTdata(self):
        TCP_pose = robot.interface.getActualTCPPose()
        self.TCPPoses.append(TCP_pose)
        TCP_force = np.asarray(robot.interface.getActualTCPForce())
        self.FTMeasurements.append(TCP_force)
        self.activeFMeasurement.append(TCP_force[2])


    def saveData2File(self, fileName):
        print("starting to save data to " + fileName + ".csv" )
        Fx = np.array([])
        Fy = np.array([])
        Fz = np.array([])
        Tx = np.array([])
        Ty = np.array([])*10**(-3)
        Tz = np.array([])
        for idx, measurement in enumerate(self.FTMeasurements):
            rotm = R.from_rotvec(self.TCPPoses[idx][3:]).as_matrix()
            force = measurement[:3]
            torque = measurement[3:]
            rotated_force = np.matmul(rotm,force)
            rotated_torque = np.matmul(rotm,torque)
            Fx = np.append(Fx, rotated_force[0])
            Fy = np.append(Fy, rotated_force[1])
            Fz = np.append(Fz, rotated_force[2])
            Tx = np.append(Tx, rotated_torque[0])
            Ty = np.append(Ty, rotated_torque[1])
            Tz = np.append(Tz, rotated_torque[2])

        dict = {'Fx': Fx, 'Fy':Fy, 'Fz':Fz, 'Tx':Tx, 'Ty':Ty, 'Tz':Tz}
        df = pd.DataFrame(dict)
        df.to_csv(fileName+".csv", index=False)
        print("succesfully stored data in " + fileName)


    def butter_lowpass(self, cutoff, fs, order=1):
        ''' Copied from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units '''
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='lowpass', analog=False, output='ba')
        return b, a





    def collisionDetection(self, _axis = axis.z):
        currentForce = self.interface.getActualTCPForce()[2]
        df = currentForce - self.prevForce
        self.prevForce = currentForce
        #print(abs(df))
        if abs(df) > self.contactThres:
            return True
        else:
            return False


    def get_pos(self):
        print("Move robot to pos")
        robot.control.teachMode()
        input("Press enter to save pos")
        robot.control.endTeachMode()
        return robot.interface.getActualTCPPose()


    def setPositions(self, desired_Orientation):

        homePos = self.get_pos()
        homePos[3:] = desired_Orientation

        startPos = self.get_pos()
        startPos[3:] = desired_Orientation

        endPos = self.get_pos()
        endPos[3:] = desired_Orientation

        f = open("positions.txt",'w')
        f.write(str(homePos)+"\n")
        f.write(str(startPos) + "\n")
        f.write(str(endPos) + "\n")
        f.close()
        return homePos, startPos, endPos


    def calcStepSizes(self, dt, dVel, beginPose, endPose, _axis=axis.z):
        nSteps = int(np.linalg.norm(np.array(beginPose[0:1]) - np.array(endPose[0:1])) / dVel / dt)

        beginPos = beginPose[0:2]
        endPos = endPose[0:2]
        distX = endPos[0] - beginPos[0]
        distY = endPos[1] - beginPos[1]
        return distX/nSteps, distY/nSteps, nSteps

    def swipe(self, pose, dPose, dVel, dForce, stepX, stepY, force_axis=axis.z):
        swipeAcc = 1.2

        cf = self.forceControl(dForce, _axis=force_axis)
        pose[3:] = pose[3:] # Keep desired pose orientation
        pose[0] += 0 #stepX
        pose[1] += 0 #stepY
        pose[2] -= cf

        robot.control.servoL(pose, dVel, swipeAcc, self.sampleTime, 0.03, 2000) #600)
        return pose

    def forceControl(self, dForce, _axis=axis.z):
        fc_f = 1 # [Hz]
        #LPF_alpha_f = calculate_lpf_alpha(fc_f, self.sampleTime)
        #filtered_force = lp_filter(self.activeFMeasurement[-1], self.activeFMeasurement[-2], LPF_alpha_f)

        #b,a = self.butter_lowpass(2, self.sampleFreq, 1)
        #filtered_force = lfilter(b, a, self.activeFMeasurement)

        filtered_force, self.z = signal.lfilter(self.b, 1, self.activeFMeasurement , zi=self.z)

        currForce = filtered_force[-1]
        deltaFT = dForce - currForce

        proportional = deltaFT * self.kF
        derivative = (deltaFT - self.prevForce) * self.dF
        integral = (self.prevIntegral + deltaFT * self.sampleTime) * self.iF
        controlSignal = proportional + derivative + integral


        test_array1.append(self.activeFMeasurement)
        test_array2.append(filtered_force[-1])
        test_array4.append(deltaFT)
        test_array5.append(integral)
        test_array3.append(controlSignal)

        # save prev deltaFT
        self.prevIntegral = integral
        self.prevForce = deltaFT

        return controlSignal


def runCollisionDetection(robot,startPos, collisionVel, collisionAcc, sampleTime):
    robot.control.moveL(startPos, collisionVel, collisionAcc, True)
    start_time = time.time()
    stop_loop = 0
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

def runReleaseForce(robot,startPos,releaseVel,releaseAcc, sampleTime,desiredForce):
    robot.control.moveL(startPos, releaseVel, releaseAcc, True)
    while robot.interface.getActualTCPForce()[2] > desiredForce:
        time.sleep(sampleTime)

    print(robot.interface.getActualTCPForce()[2])
    robot.control.stopL(10.0)

def runForceController(robot, desiredVel, startPos, endPos, sampleTime, desiredForce):
    stepX, stepY, nSteps = robot.calcStepSizes(sampleTime, desiredVel, startPos, endPos, axis.z)
    nextPose = robot.interface.getActualTCPPose()  # startPos # TODO: Should be startPos <-- Only for test
    #print("stepX ",stepX,"stepY ",stepY,"nSteps ",nSteps)
    #fig, ax = plt.subplots(5)
    try:
        #for i in range(nSteps):
        while(True):
            start = time.time()
            temp = robot.interface.getActualTCPPose()
            test_array6.append(temp[:3])
            robot.collectRawFTdata()
            nextPose = robot.swipe(nextPose, endPos, desiredVel, desiredForce, stepX, stepY, axis.z)
            dur = time.time() - start
            if dur < sampleTime:
                time.sleep(sampleTime - dur)
    except KeyboardInterrupt:
        robot.control.servoStop()
        robot.control.speedStop()

        fig, ax = plt.subplots(5)
        ax[0].plot(range(len(test_array1[-1])),test_array1[-1]) # unfiltered force
        #ax[1].plot(range(len(test_array2[-1])),test_array2[-1]) # filtered force
        ax[1].plot(range(len(test_array2)),test_array2) # filtered force
        ax[2].plot(range(len(test_array3[1:])),test_array3[1:])         # controlSignal
        ax[3].plot(range(len(test_array4)),test_array4)         # deltaFT
        ax[4].plot(range(len(test_array6)), test_array6,label = ['x','y','z'])
        plt.show()

    fig, ax = plt.subplots(5)
    ax[0].plot(range(len(test_array1[-1])), test_array1[-1])    # unfiltered force
    #ax[1].plot(range(len(test_array2[-1])), test_array2[-1]) # filtered force
    ax[1].plot(range(len(test_array2)),test_array2) # filtered force
    ax[2].plot(range(len(test_array3[1:])), test_array3[1:])    # controlSignal
    ax[3].plot(range(len(test_array4)), test_array4)            # deltaFT
    #ax[4].plot(range(len(test_array5)), test_array5)            # integral
    ax[4].plot(range(len(test_array6)), test_array6,label = ['x','y','z'])
                # TCP
    plt.show()

if __name__ == '__main__':
    ip = "192.168.1.130" # ip for main
    #ip = "192.168.1.111" # ip for secondary

    sampleFreq = 500  # [Hz]
    robot = Robot(ip, sampleFreq)
    sampleTime = 1/sampleFreq

    desiredOrientation = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec()
    desiredVel = 0.1   # [m/s]
    desiredAcc = 0.5 #1.2  # [m/s^2]
    desiredForce = 1 # [N] (1.8N is almost equal to the real zero)

    # Collision detection
    collisionVel = 0.05  # [m/s]
    collisionAcc = 0.5  # [m/s^2]

    # Release force
    releaseVel = 0.001  # [m/s]
    releaseAcc = 0.5  # [m/s^2]

    # Resetting sensors
    robot.control.zeroFtSensor()

    # Set positions
    homePos, startPos, endPos = robot.setPositions(desiredOrientation)

    robot.control.moveL(homePos)

    startPos[2] -= 0.1  # [m] offset

    # Starting collision detection
    print("Run collision detection")
    #runCollisionDetection(robot, startPos, collisionVel, collisionAcc, sampleTime)
    input("Press Enter to continue.")

    # Release force
    startPos[2] += 0.1  # [m]
    #runReleaseForce(robot, startPos, releaseVel, releaseAcc, sampleTime, desiredForce)
    print("Force released")
    input("Press Enter to continue.")


    # Starting force controller
    print("Running force controller")
    runForceController(robot, desiredVel, startPos, endPos, sampleTime, desiredForce)

    print("Ended program!")
