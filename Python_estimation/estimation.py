import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import Bounds
from random import uniform
import os
from scipy.spatial.transform import Rotation as R
from scipy import signal

import time

def main(forces_tcp, moments_tcp,vel_tcp):

    r = 31 * pow(10,-3) # radius of finder [m]
    A = np.eye(3)
    dz = 0


    length = forces_tcp.shape[1]

    F_normals = np.zeros((length,1))
    F_frictions = np.zeros((length,1))

    for i in range(length):
        f = forces_tcp[:,i]
        m = moments_tcp[:,i]

        [c_sph, k_sph] = contactCentroidEstimation(A, r, f, m)

        x0 = np.hstack((np.array([0, 0, r/2]),k_sph))
        xdata = np.hstack((f,m))

        #res = least_squares(G, x0, method='trf', bounds=([0,0,0,-np.inf], [r,r,r,np.inf]), verbose=0, args=xdata)
        res = least_squares(G, x0, method='lm', xtol=1e-8, verbose=0, args=xdata)

        F_normal, F_friction = estimateFrictionForces(res.x[0:3], f)
        F_normals[i] = np.linalg.norm(F_normal)
        F_frictions[i] = np.linalg.norm(F_friction)

    min_idx, max_idx = startIndex(F_frictions)

    tic = time.perf_counter()
    x = estimateFrictionCoefficients(F_normals, vel_tcp, F_frictions, min_idx, max_idx)
    toc = time.perf_counter()

    return x

def jac(x0, F_n, v, F):
    #J = np.empty((4,len(F)))
    mu_s = x0[0]
    mu_c = x0[1]
    v0 = x0[2]
    nabla = x0[3]

    J = [np.exp(-pow(v,2)/pow(v0,2)), 1 - np.exp(-pow(v,2)/pow(v0,2)), -(2*pow(v,2)*np.exp(-pow(v,2)/pow(v0,2))*(mu_c - mu_s))/pow(v0,3), v/F_n]

    # Reshape
    J = np.array(J).reshape((4,400))
    J = np.transpose(J)

    return J

def estimateFrictionCoefficients(F_normals, vel_tcp, F_frictions, min_idx, max_idx):
    x = [0, 0, 0, 0]
    x_prev = 0
    x_diff = []
    x_coeffs1 = []
    x_coeffs2 = []
    x_coeffs3 = []
    x_coeffs4 = []
    optimality = []
    cost = []
    gradient = []

    for n_start in range(min_idx, max_idx + 1, 2):
        bool = True

        n_data = 400 #400
        xdata = [F_normals[n_start:n_start + n_data], vel_tcp[n_start:n_start + n_data],F_frictions[n_start:n_start + n_data]]

        bounds = ([0.1, 0.1, 0, 0], [2, 2, 10, 0.1])

        cnt = 0
        while bool:
            x0 = []
            for i in range(4):
                x0.append(np.random.uniform(bounds[0][i], bounds[1][i]))

            res = least_squares(g_func, x0=x0, bounds=bounds, method='trf', jac=jac, x_scale='jac', verbose=0, args=xdata, ftol=1e-8, xtol=1e-8, gtol=1e-8)

            # Changing the threshold dramatically decreases the computational time, however it might invalidate the result
            if res.optimality < 1e-01:
                bool = False
            cnt += 1
            #print(cnt)
            if cnt > 1000:
                bool = False

        x = x + res.x

        x_diff.append(np.linalg.norm(res.x-x_prev))
        x_coeffs1.append(res.x[0])
        x_coeffs2.append(res.x[1])
        x_coeffs3.append(res.x[2])
        x_coeffs4.append(res.x[3])
        x_prev = res.x
        optimality.append(res.optimality)
        cost.append(res.cost)
        gradient.append(np.linalg.norm(res.grad))
    #return x/(max_idx-min_idx+1)
    return x/(int(max_idx-min_idx)/2+1)

def diff(array):

    diff = np.zeros(len(array))
    for i in range(len(array)-1):
        diff[i] = array[i+1] - array[i]
    return diff


def startIndex(F_frictions):
    diffs = diff(F_frictions.flatten())
    cnt = 0

    while abs(diffs[cnt]) > 0.01:
        cnt = cnt + 1
    while diffs[cnt] <= 0.01:
        cnt = cnt + 1
    min_idx = cnt
    while diffs[cnt] > 0.01:
        cnt = cnt + 1
    max_idx = cnt

    return min_idx, max_idx

def contactCentroidEstimation(A,R,f,m):
    # Sphere force centroid estimation_ single Old Paper
    sig_prime = pow(np.linalg.norm(m),2)-pow(R,2)*pow(np.linalg.norm(f),2)
    K_sphere = -np.sign(np.dot(np.matrix.transpose(f),m)) / (np.sqrt(2)*R) * np.sqrt(sig_prime+np.sqrt(pow(sig_prime,2)+4*pow(R,2)*pow(np.dot(np.matrix.transpose(f),m),2)))


    if K_sphere > 0:
        c_sphere = (1)/(K_sphere*(pow(K_sphere,2)+pow(np.linalg.norm(f),2)))*(pow(K_sphere,2)*m+K_sphere*np.cross(f,m)+(np.transpose(f)*m)*f)
    else:
        r_0 = np.cross(f,m)/pow(np.linalg.norm(f),2)
        f_prime = np.dot(A,f)
        r_0_prime = np.dot(A,r_0)

        lambda1 = np.dot(-np.transpose(f_prime),r_0_prime)
        lambda2 = np.sqrt(pow(np.dot(np.transpose(f_prime),r_0_prime),2)-pow(np.linalg.norm(f_prime),2)*(pow(np.linalg.norm(r_0_prime),2)-pow(R,2)))
        lambda3 = pow(np.linalg.norm(f_prime),2)
        lambda_ = (lambda1-lambda2) / lambda3
        #lambda_ = (-np.transpose(f_prime)*r_0_prime-np.sqrt(pow(np.transpose(f_prime)*r_0_prime,2)-pow(np.linalg.norm(f_prime),2)*pow(np.linalg.norm(r_0_prime),2)-pow(R,2))) / pow(np.linalg.norm(f_prime),2)
        c_sphere = r_0 + lambda_ * f
    return [c_sphere, K_sphere]


def S(p):
    dz = 0
    r = 31*pow(10,-3)
    return pow(p[0], 2) + pow(p[1], 2) + pow((p[2]-dz), 2) - pow(r, 2)


def G(x0, xdata1, xdata2, xdata3, xdata4, xdata5, xdata6):
    xdata = np.array((xdata1, xdata2, xdata3, xdata4, xdata5, xdata6))

    # Cannot take the gradient to a scalar?
    #y[1,1] = np.cross(xdata[0:3], x0[0:3]) + x0[3]*np.gradient(S(x0[0:3])) - xdata[4:6]
    y1 = np.cross(xdata[0:3], x0[0:3]) - xdata[3:6]
    y2 = S(x0[0:3])
    y = np.hstack((y1,y2))
    return y


def estimateFrictionForces(P_c, F):
    Q = 2*P_c
    a = (np.dot(np.transpose(Q),F))/(np.dot(np.transpose(Q),Q))

    F_normal = np.dot(a,Q)
    cos_theta = np.linalg.norm(F_normal)/np.linalg.norm(F)

    if 1-pow(cos_theta,2) <= 0:
        F_friction = [0,0,0]
    else:
        F_friction = (np.sqrt(1-pow(cos_theta,2))/cos_theta) * F_normal

    return F_normal, F_friction


def g_func(x0, Fn, v, F):
    g_mat = np.zeros((len(Fn), 1))
    mu_s = x0[0]
    mu_c = x0[1]
    v0 = x0[2]
    nabla = x0[3]

    for i in range(len(Fn)):
        g_mat[i] = mu_c + (mu_s - mu_c) * np.exp(-pow(v[i] / v0, 2)) + nabla * (v[i] / Fn[i]) - F[i] / Fn[i]

    return g_mat.flatten()

def classification_data():
    path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Python_implementation/"
    data = ["hard_steel_on_hard_steel", "Mild_steel_on_mild_steel", "Mild_steel_on_lead", "Aluminum_on_mild_steel",
            "Copper_on_mild_steel", "Nickel_on_nickel", "Brass_on_mild_steel", "Zinc_on_cast_iron",
            "Copper_on_cast_iron", "Aluminum_on_aluminum", "Glass_on_glass", "Oak_on_oak_(parallel_to_grain)",
            "Oak_on_oak_(perpendicular)", "Leather_on_oak_(parallel)"]

    mu_static = []
    mu_dyn = []
    strib = []
    visc = []
    coefficients = []
    labels = []
    iterations_pr_sample = 10

    for i in range(len(data)):
    #for i in range(2):
        print("Iteration: ", i)
        for j in range(iterations_pr_sample):
        #for j in range(2):
            forces_tcp = scipy.io.loadmat(path+"forces_tcp_"+data[i] + ".mat")
            moments_tcp = scipy.io.loadmat(path + "moments_tcp_" + data[i] + ".mat")
            vel_tcp = scipy.io.loadmat(path + "vel_tcp_" + data[i] + ".mat")

            forces_tcp = forces_tcp['forces_tcp']
            moments_tcp = moments_tcp['moments_tcp']
            vel_tcp = vel_tcp['vel_tcp']

            # Noise
            for k in range(len(forces_tcp)):
                forces_tcp[k] += np.random.normal(0, 0.05)
                moments_tcp[k] += np.random.normal(0, 0.3e-4)
                vel_tcp[k] += np.random.normal(0, 0.05)

            x = main(forces_tcp, moments_tcp, vel_tcp)
            mu_static.append(x[0])
            mu_dyn.append(x[1])
            strib.append(x[2])
            visc.append(x[3])
            coefficients.append(x)
            labels.append(i)

    return np.asarray(coefficients), np.asarray(labels)

def test_func():
    path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Python_implementation/"
    data = ["hard_steel_on_hard_steel","Mild_steel_on_mild_steel", "Mild_steel_on_lead", "Aluminum_on_mild_steel", "Copper_on_mild_steel","Nickel_on_nickel", "Brass_on_mild_steel","Zinc_on_cast_iron", "Copper_on_cast_iron", "Aluminum_on_aluminum", "Glass_on_glass", "Oak_on_oak_(parallel_to_grain)", "Oak_on_oak_(perpendicular)","Leather_on_oak_(parallel)"]
    mu_static_GT = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10, 0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61]
    mu_dyn_GT =    [0.42, 0.57, 0.95, 0.47, 0.36, 0.53, 0.44, 0.21, 0.29, 1.40, 0.40, 0.48, 0.32, 0.52]

    mu_static = []
    mu_dyn = []
    strib = []
    visc = []
    error_static = []
    error_dyn = []

    for i in range(len(mu_dyn_GT)):
        print("Iteration: ", i)
        forces_tcp = scipy.io.loadmat(path+"forces_tcp_"+data[i] + ".mat")
        moments_tcp = scipy.io.loadmat(path + "moments_tcp_" + data[i] + ".mat")
        vel_tcp = scipy.io.loadmat(path + "vel_tcp_" + data[i] + ".mat")

        forces_tcp = forces_tcp['forces_tcp']
        moments_tcp = moments_tcp['moments_tcp']
        vel_tcp = vel_tcp['vel_tcp']

        #Noise
        #for i in range(len(forces_tcp)):
            #forces_tcp[i] += np.random.normal(0, 0.001)
            #moments_tcp[i] += np.random.normal(0, 1e-5)
            #vel_tcp[i] += np.random.normal(0, 0.01)

        x = main(forces_tcp, moments_tcp, vel_tcp)
        mu_static.append(x[0])
        mu_dyn.append(x[1])
        strib.append(x[2])
        visc.append(x[3])
        error_static.append(x[0] - mu_static_GT[i])
        error_dyn.append(x[1] - mu_dyn_GT[i])


    plt.figure(10)
    plt.plot(range(len(mu_static)), error_static, label="mu_static error")
    plt.plot(range(len(mu_dyn)), error_dyn, label="mu_dynamic error")
    plt.legend()
    #plt.show()
    return 0

def filter(FMeasurement):
    #Filter setup
    fc_f = 1 # [Hz]
    b = signal.firwin(numtaps = 50, cutoff= fc_f, fs = 500) # run once
    z = signal.lfilter_zi(b, 1) # run once
    filtered_force, z = signal.lfilter(b, 1, FMeasurement, zi=z) # run me

    return filtered_force

def gen_surface_properties_from_folder():
    path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/data_RW_pap/"
    dir =  "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/"


    idx = 0
    cnt = 0
    for folder in os.listdir(path):
        mu_static = []
        mu_dyn = []
        strib = []
        visc = []
        coefficients = []
        labels = []
        for filename in os.listdir(path + folder + '/'):
            print('cnt: ', cnt)
            print('label: ', idx)
            print()
            cnt += 1
            with open(path + folder + '/' + filename, 'rb') as f:
                data = np.load(f)

            forces_tcp = np.transpose(data[0, :, 0:3])
            moments_tcp = np.transpose(data[0, :, 3:6])
            vel_tcp = []

            # Transform from base to tcp
            rotm = R.from_rotvec(data[2, :, 3:6]).as_matrix()
            for i in range(forces_tcp.shape[1]):
                forces_tcp[:, i] = np.matmul(forces_tcp[:, i], rotm[i, :, :])
                moments_tcp[:, i] = np.matmul(moments_tcp[:, i], rotm[i, :, :])

            forces_tcp[0, :] = filter(forces_tcp[0, :])
            forces_tcp[1, :] = filter(forces_tcp[1, :])
            forces_tcp[2, :] = filter(forces_tcp[2, :])

            moments_tcp[0, :] = filter(moments_tcp[0, :])
            moments_tcp[1, :] = filter(moments_tcp[1, :])
            moments_tcp[2, :] = filter(moments_tcp[2, :])

            #print(forces_tcp.shape)
            #plt.plot(forces_tcp[2, :])
            #plt.show()

            for i in range(forces_tcp.shape[1]):
                vel_tcp.append(np.linalg.norm(data[1, i, 0:3]))

            vel_tcp = np.asarray(vel_tcp).reshape((forces_tcp.shape[1], 1))

            x = main(forces_tcp, moments_tcp, vel_tcp)
            mu_static.append(x[0])
            mu_dyn.append(x[1])
            strib.append(x[2])
            visc.append(x[3])
            coefficients.append(x)
            labels.append(idx)
        idx += 1
        with open(dir + 'data_RW_estimation/' + 'coeff_' + folder + '.npy', 'wb') as f:
            np.save(f, np.asarray(coefficients))
        with open(dir + 'data_RW_estimation/' + 'labels_' + folder + '.npy', 'wb') as f:
            np.save(f, np.asarray(labels))


    return np.asarray(coefficients), np.asarray(labels)

'''
forces_tcp = scipy.io.loadmat('C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Python_implementation/forces_tcp_aluminum_on_aluminum.mat')
moments_tcp = scipy.io.loadmat('C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Python_implementation/moments_tcp_aluminum_on_aluminum.mat')
vel_tcp = scipy.io.loadmat('C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Python_implementation/vel_tcp_aluminum_on_aluminum.mat')

forces_tcp = forces_tcp['forces_tcp']
moments_tcp = moments_tcp['moments_tcp']
vel_tcp = vel_tcp['vel_tcp']

tic = time.perf_counter()
#main(forces_tcp, moments_tcp, vel_tcp)
#test_func()
coefficients, labels = classification_data()

print('labels: ', labels)



np.savetxt('labels.csv', np.array(labels).astype(int), delimiter=',')
np.savetxt('coefficients.csv', np.array(coefficients), delimiter=',')



toc = time.perf_counter()


print("Elapsed time of main: ", toc-tic)
plt.show()
'''