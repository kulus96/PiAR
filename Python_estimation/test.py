import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
import os
from scipy.spatial.transform import Rotation as R
from scipy import signal
import scipy


sys.path.append('../Python_classification')
from classification import test_of_classifier
import estimation as est


def surface_properties_data_collection():

    #Estimate surface properties
    coeff, labels = est.classification_data()

    with open('Results/estimation_coeff_noise_results.npy', 'wb') as f:
      np.save(f, coeff)
    with open('Results/estimation_labels_noise_results.npy', 'wb') as f:
      np.save(f, labels)


def classification_test():
    # Open estimation properties
    path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/Simulation/"

    with open(path + 'estimation_coeff_results.npy', 'rb') as f:
        coeff = np.load(f)
    with open(path + 'estimation_labels_results.npy', 'rb') as f:
        labels = np.load(f)
    with open(path + 'estimation_coeff_noise_results.npy', 'rb') as f:
        coeff_noise = np.load(f)
    with open(path + 'estimation_labels_noise_results.npy', 'rb') as f:
        labels_noise = np.load(f)

    # Classification
    methods = ["GNB", "SVM_poly", "SVM_linear", "k-nearest neighbors", "LDA", "DTC", "AdaBoostClassifier", "RandomForest", "GPC", "MLP"]
    methods_short = ["GNB", "SVM_poly", "SVM_linear", "KNN", "LDA", "DTC", "AdaBoost", "RandomForest", "GPC", "MLP"]
    #methods = ["SVM_poly"]

    means = []
    means_noise = []

    for names in methods:
        scores, mean, std, total_points, mislabeled_points = test_of_classifier(coeff, labels, names, test_data_size=0.3)
        means.append(mean)
        scores, mean, std, total_points, mislabeled_points = test_of_classifier(coeff_noise, labels_noise, names, test_data_size=0.3)
        means_noise.append(mean)
        #plt.plot(scores)


    # Save means
    with open(path + 'classification_means_results.npy', 'wb') as f:
      np.save(f, means)
    with open(path + 'classification_means_noise_results.npy', 'wb') as f:
      np.save(f, means_noise)

    np_to_csv(path + 'classification_means_results.npy')
    np_to_csv(path + 'classification_means_noise_results.npy')

    my_dpi = 100
    plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    # Set position of bar on X axis
    barWidth = 0.33
    br1 = np.arange(len(methods))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, means, color='b', width=barWidth, align='center', label='without noise')
    plt.bar(br2, means_noise, color='r', width=barWidth, align='center', label='with noise')
    plt.xticks(rotation=45)
    plt.xticks([r + barWidth - 0.33/2 for r in range(len(methods))], methods_short)
    r = range(len(methods))
    for i in range(len(means)):
        plt.text(r[i] - barWidth/2 , means[i] + 0.01, round(means[i],2), color='b')
        plt.text(r[i] + barWidth/2 + 0.05, means_noise[i] + 0.01, round(means_noise[i], 2), color='r')
    plt.ylabel('Accuracy')
    plt.legend('lower right')
    plt.savefig('Results/classification_on_simulated_data')
    plt.show()

    print(means)
    print(means_noise)


def surface_properties_test():
    # Open estimation properties
    with open('Results/estimation_coeff_results.npy', 'rb') as f:
        coeff = np.load(f)
    with open('Results/estimation_labels_results.npy', 'rb') as f:
        labels = np.load(f)
    with open('Results/estimation_coeff_noise_results.npy', 'rb') as f:
        coeff_noise = np.load(f)
    with open('Results/estimation_labels_noise_results.npy', 'rb') as f:
        labels_noise = np.load(f)

    GT_mu_s = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10, 0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61]
    GT_mu_c = [0.42, 0.57, 0.95, 0.47, 0.36, 0.53, 0.44, 0.21, 0.29, 1.4, 0.40, 0.48, 0.32, 0.52]

    print(coeff)
    print(coeff[:,0])

    plt.scatter(GT_mu_s, GT_mu_c, color='b')
    plt.scatter(coeff[:, 0], coeff[:, 1], color='r')
    plt.scatter(coeff_noise[:, 0], coeff_noise[:, 1])
    plt.show()

def filter(FMeasurement):
    #Filter setup
    fc_f = 1 # [Hz]
    b = signal.firwin(numtaps = 50, cutoff= fc_f, fs = 500) # run once
    z = signal.lfilter_zi(b, 1) # run once
    filtered_force, z = signal.lfilter(b, 1, FMeasurement, zi=z) # run me

    return filtered_force

def G(x0, xdata1, xdata2, xdata3, xdata4, xdata5, xdata6):
    xdata = np.array((xdata1, xdata2, xdata3, xdata4, xdata5, xdata6))

    # Cannot take the gradient to a scalar?
    #y[1,1] = np.cross(xdata[0:3], x0[0:3]) + x0[3]*np.gradient(S(x0[0:3])) - xdata[4:6]
    y1 = np.cross(xdata[0:3], x0[0:3]) - xdata[3:6]
    y2 = S(x0[0:3])
    y = np.hstack((y1,y2))
    return y

def contact_centroid_estimation_simulation():
    path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Python_implementation/"
    dir = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/Simulation/"
    data = ["hard_steel_on_hard_steel", "Mild_steel_on_mild_steel", "Mild_steel_on_lead", "Aluminum_on_mild_steel",
            "Copper_on_mild_steel", "Nickel_on_nickel", "Brass_on_mild_steel", "Zinc_on_cast_iron",
            "Copper_on_cast_iron", "Aluminum_on_aluminum", "Glass_on_glass", "Oak_on_oak_(parallel_to_grain)",
            "Oak_on_oak_(perpendicular)", "Leather_on_oak_(parallel)"]
    mu_static_GT = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10, 0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61]
    mu_dyn_GT = [0.42, 0.57, 0.95, 0.47, 0.36, 0.53, 0.44, 0.21, 0.29, 1.40, 0.40, 0.48, 0.32, 0.52]

    mu_static = []
    mu_dyn = []
    strib = []
    visc = []
    error_static = []
    error_dyn = []
    centroids = []
    centroids_all = []
    K_spheres = []

    for i in range(len(mu_dyn_GT)):
        print("Iteration: ", i)
        forces_tcp = scipy.io.loadmat(path + "forces_tcp_" + data[i] + ".mat")
        moments_tcp = scipy.io.loadmat(path + "moments_tcp_" + data[i] + ".mat")
        vel_tcp = scipy.io.loadmat(path + "vel_tcp_" + data[i] + ".mat")

        forces_tcp = forces_tcp['forces_tcp']
        moments_tcp = moments_tcp['moments_tcp']
        vel_tcp = vel_tcp['vel_tcp']

        centroid = [0, 0, 0]

        r = 31e-3
        A = np.eye(3)

        c_sphere, K_sphere = est.contactCentroidEstimation(A, r, forces_tcp[:, 0], moments_tcp[:, 0])

        x0 = np.hstack((np.array([0, 0, r]), K_sphere))


        for i in range(np.asarray(forces_tcp).shape[1]):
            f = forces_tcp[:, i]
            m = moments_tcp[:, i]

            c_sphere, K_sphere = est.contactCentroidEstimation(A, r, f, m)
            xdata = np.hstack((f, m))

            bounds_min = [(-r, r), (-r, r), (0, r), (-np.inf, np.inf)]
            # res = least_squares(est.G, x0, method='trf', bounds=([0,0,0,-np.inf], [r,r,r,np.inf]), verbose=0, args=xdata)
            res = minimize(est.G, x0, args=(xdata[0], xdata[1], xdata[2], xdata[3], xdata[4], xdata[5]),
                           bounds=bounds_min, constraints={'type': 'eq', 'fun': est.S})
            centroid.append(res.x[0:3])
            #x0[0:3] = res.x[0:3]
            x0 = res.x
            K_spheres.append(K_sphere)
            centroids_all.append(res.x[0:3])



        # centroids_external.append(np.sum(centroid_external,axis=0)/len(forces_tcp))
        # centroids.append(np.sum(centroid,axis=0)/len(forces_tcp))
        centroids.append(res.x[0:3])
        #centroids_labels.append(filename)

    centroids = np.asarray(centroids)
    K_spheres = np.asarray(K_spheres)
    centroids_all = np.asarray(centroids_all)
    #centroids_labels = np.asarray(centroids_labels)



    with open(dir + 'contact_centroid_sim' + '.npy', 'wb') as f:
        np.save(f, centroids)
    with open(dir + 'contact_centroid_K_values' + '.npy', 'wb') as f:
        np.save(f, K_spheres)
    with open(dir + 'contact_centroids_all_data' + '.npy', 'wb') as f:
        np.save(f, centroids_all)

    np_to_csv(dir + 'contact_centroid_sim' + '.npy')
    np_to_csv(dir + 'contact_centroid_K_values' + '.npy')
    np_to_csv(dir + 'contact_centroids_all_data' + '.npy')

    plt.figure(1)
    plt.plot(range(len(centroids)), centroids[:, 0], color='r')
    plt.plot(range(len(centroids)), centroids[:, 1], color='b')
    plt.plot(range(len(centroids)), centroids[:, 2], color='g')

    plt.figure(2)
    plt.plot(K_spheres)
    plt.show()

    return 0

def contact_centroid_estimation_test():
    # Open
    #path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/RW_data/Centroid_estimation_data/Alu_ori_pi4/"
    #dir = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/RW_data/Centroids_Aluminium/"
    path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/RW_data/New_Centroid_test/pi/"
    dir = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/RW_data/New_Centroid_test_results/"

    idx = 0
    cnt = 0
    centroids = []
    centroids_labels = []
    centroids_external = []
    for folder in os.listdir(path):
        for filename in os.listdir(path + folder + '/'):
            cnt += 1
            with open(path + folder + '/' + filename, 'rb') as f:
                data = np.load(f)

            forces_tcp = np.transpose(data[0, :, 0:3])
            moments_tcp = np.transpose(data[0, :, 3:6])
            forces_tcp_external = np.transpose(data[3, :, 0:3])
            moments_tcp_external = np.transpose(data[3, :, 3:6])

            # Transform from base to tcp
            # rotm = R.from_rotvec(data[2, :, 3:6]).as_matrix()
            # for i in range(forces_tcp.shape[1]):
            #     forces_tcp[:, i] = np.matmul(forces_tcp[:, i], rotm[i, :, :])
            #     moments_tcp[:, i] = np.matmul(moments_tcp[:, i], rotm[i, :, :])
            #
            # # # Apply filter
            # forces_tcp[0, :] = filter(forces_tcp[0, :])
            # forces_tcp[1, :] = filter(forces_tcp[1, :])
            # forces_tcp[2, :] = filter(forces_tcp[2, :])
            # moments_tcp[0, :] = filter(moments_tcp[0, :])
            # moments_tcp[1, :] = filter(moments_tcp[1, :])
            # moments_tcp[2, :] = filter(moments_tcp[2, :])
            #
            forces_tcp_external[0, :] = filter(forces_tcp_external[0, :])
            forces_tcp_external[1, :] = filter(forces_tcp_external[1, :])
            forces_tcp_external[2, :] = filter(forces_tcp_external[2, :])
            moments_tcp_external[0, :] = filter(moments_tcp_external[0, :])
            moments_tcp_external[1, :] = filter(moments_tcp_external[1, :])
            moments_tcp_external[2, :] = filter(moments_tcp_external[2, :])

            centroid = [0, 0, 0]
            centroid_external = [0, 0, 0]
            r = 31e-3
            A = np.eye(3)

            c_sphere, K_sphere = est.contactCentroidEstimation(A, r, forces_tcp[:,0], moments_tcp[:,0])

            x0 = np.hstack((np.array([0, 0, r / 2]), K_sphere))
            x0_external = np.hstack((np.array([0, 0, r / 2]), K_sphere))

            n_down_sample = 10
            forces_tcp = forces_tcp[:, ::n_down_sample]
            moments_tcp = moments_tcp[:, ::n_down_sample]
            forces_tcp_external = forces_tcp_external[:, ::n_down_sample]
            moments_tcp_external = moments_tcp_external[:, ::n_down_sample]


            for i in range(forces_tcp.shape[1]):

                f = forces_tcp[:, i]
                m = moments_tcp[:, i]
                f_external = forces_tcp_external[:, i]
                m_external = moments_tcp_external[:, i]


                c_sphere, K_sphere = est.contactCentroidEstimation(A,r,f,m)
                c_sphere_external, K_sphere_external = est.contactCentroidEstimation(A, r, f_external, m_external)



                xdata = np.hstack((f, m))
                xdata_external = np.hstack((f_external, m_external))
                bounds_min = [(0,r), (0,r), (0,r), (-np.inf, 0)]

                #res = least_squares(est.G, x0, method='trf', bounds=([0,0,0,-np.inf], [r,r,r,np.inf]), verbose=0, args=xdata)
                res = minimize(est.G, x0, args=(xdata[0], xdata[1], xdata[2], xdata[3], xdata[4], xdata[5]), bounds=bounds_min, constraints={'type':'eq', 'fun':est.S})


                #res_external = least_squares(est.G, x0_external, method='trf', bounds=([0, 0, 0, -np.inf], [r, r, r, np.inf]), verbose=0,args=xdata_external)
                res_external = minimize(est.G, x0_external, args=(xdata_external[0], xdata_external[1], xdata_external[2], xdata_external[3], xdata_external[4], xdata_external[5]),
                               bounds=bounds_min, constraints={'type': 'eq', 'fun': est.S})

                centroid.append(res.x[0:3])
                centroid_external.append(res_external.x[0:3])


                x0[0:3] = res.x[0:3]
                x0_external[0:3] = res_external.x[0:3]
                #0 = res.x
                #x0_external = res_external.x

            #centroids_external.append(np.sum(centroid_external,axis=0)/len(forces_tcp))
            #centroids.append(np.sum(centroid,axis=0)/len(forces_tcp))
            centroids.append(res.x[0:3])
            centroids_external.append(res_external.x[0:3])
            centroids_labels.append(filename)



    centroids = np.asarray(centroids)
    centroids_labels = np.asarray(centroids_labels)
    centroids_external = np.asarray(centroids_external)

    print(centroids)

    print('centroids shape: ', centroids.shape)
    print('centroids ext shape: ', centroids_external.shape)

    # Save data
    with open(dir + 'ori_pi4_centroids_internal' + '.npy', 'wb') as f:
        np.save(f, centroids)
    with open(dir + 'ori_pi4_centroids_external' + '.npy', 'wb') as f:
        np.save(f, centroids_external)
    with open(dir + 'ori_pi4_labels' + '.npy', 'wb') as f:
        np.save(f, centroids_labels)

    #np_to_csv(dir + 'ori_pi4_centroids_internal.npy')
    #np_to_csv(dir + 'ori_pi4_centroids_external.npy')
    #np_to_csv(dir + 'ori_pi4_labels.npy')

    print(centroids_external.shape)
    plt.figure(1)
    plt.plot(range(len(centroids)), centroids[:, 0], color='r')
    plt.plot(range(len(centroids)), centroids[:, 1], color='b')
    plt.plot(range(len(centroids)), centroids[:, 2], color='g')

    plt.figure(2)
    plt.plot(range(len(centroids_external)), centroids_external[:, 0], color='r')
    plt.plot(range(len(centroids_external)), centroids_external[:, 1], color='b')
    plt.plot(range(len(centroids_external)), centroids_external[:, 2], color='g')

    centroid_external = np.asarray(centroid_external)
    print(centroid_external.shape)


    plt.figure(3)
    plt.plot(range(len(centroid_external)), centroid_external[:, 0], color='r')
    plt.plot(range(len(centroid_external)), centroid_external[:, 1], color='b')
    plt.plot(range(len(centroid_external)), centroid_external[:, 2], color='g')
    plt.show()



def surface_properties_data_test():
    # Estimate surface properties
    #coeff, labels = est.gen_surface_properties_from_folder()
    #
    # with open('Results/data3_RW_estimation_coeff_results.npy', 'wb') as f:
    #     np.save(f, coeff)
    # with open('Results/data3_RW_estimation_labels_results.npy', 'wb') as f:
    #     np.save(f, labels)

    with open('Results/data_RW_estimation/coeff_F10_vel_0-1_ori_pi.npy', 'rb') as f:
      coeff = np.load(f)
    #with open('Results/data2_RW_estimation_labels_results.npy', 'rb') as f:
    #  labels = np.load(f)

    print(coeff)
    print(coeff.shape)
    plt.plot(coeff[:, 1])
    plt.show()


def np_to_csv(file_name):
    #file_name = str(sys.argv[1])

    #npy_file = np.load(file_name + '.npy')
    npy_file = np.load(file_name)

    np.savetxt(file_name + '.csv', npy_file, delimiter=',')


path = "C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/RW_data/Coefficients_new/"
np_to_csv(path + 'coeff_Data_All.npy')
np_to_csv(path + 'coeff_Data_Pap.npy')
np_to_csv(path + 'coeff_Data_Alu.npy')


#surface_properties_data_test()
#contact_centroid_estimation_test()
#contact_centroid_estimation_simulation()
#est.gen_surface_properties_from_folder()
#classification_test()

#contact_centroid_estimation_simulation()



# path = 'C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/Results/RW_data/Coefficients/ALUMINIUM/coeffs/'
#
# for filename in os.listdir(path):
#     np_to_csv(path + filename)


