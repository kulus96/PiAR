import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

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
    with open('Results/estimation_coeff_results.npy', 'rb') as f:
        coeff = np.load(f)
    with open('Results/estimation_labels_results.npy', 'rb') as f:
        labels = np.load(f)
    with open('Results/estimation_coeff_noise_results.npy', 'rb') as f:
        coeff_noise = np.load(f)
    with open('Results/estimation_labels_noise_results.npy', 'rb') as f:
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
    with open('Results/Classification/classification_means_results.npy', 'wb') as f:
      np.save(f, coeff)
    with open('Results/Classification/classification_means_noise_results.npy', 'wb') as f:
      np.save(f, labels)

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

def contact_centroid_estimation_test():
    # Open
    with open('Results/20_desiredForce_Ang_3.141592653589793_0_0.npy', 'rb') as f:
        data = np.load(f)

    centroids = []
    for i in range(len(data[0])):
        f = data[0, i, 0:3]
        m = data[0, i, 3:6]
        r = 31e-3
        A = np.eye(3)
        c_sphere, K_sphere = est.contactCentroidEstimation(A,r,f,m)

        x0 = np.hstack((np.array([0, 0, r / 2]), K_sphere))
        xdata = np.hstack((f, m))

        # res = least_squares(G, x0, method='trf', bounds=([0,0,0,-np.inf], [r,r,r,np.inf]), verbose=0, args=xdata)
        res = least_squares(est.G, x0, method='lm', xtol=1e-8, verbose=0, args=xdata)
        centroids.append(res.x[0:3])

    print(centroids)

def np_to_csv(file_name):
    #file_name = str(sys.argv[1])

    npy_file = np.load(file_name + '.npy')

    np.savetxt(file_name + '.csv', npy_file, delimiter=',')

def surface_properties_data_test():
    # Estimate surface properties
    coeff, labels = est.gen_surface_properties_from_folder()
    #
    # with open('Results/data3_RW_estimation_coeff_results.npy', 'wb') as f:
    #     np.save(f, coeff)
    # with open('Results/data3_RW_estimation_labels_results.npy', 'wb') as f:
    #     np.save(f, labels)


    #with open('Results/data2_RW_estimation_coeff_results.npy', 'rb') as f:
    #   coeff = np.load(f)
    #with open('Results/data2_RW_estimation_labels_results.npy', 'rb') as f:
    #   labels = np.load(f)

    print(coeff.shape)
    plt.plot(coeff[:, 1])
    plt.show()



    # # Classification
    # methods = ["GNB", "SVM_poly", "SVM_linear", "k-nearest neighbors", "LDA", "DTC", "AdaBoostClassifier",
    #            "RandomForest", "GPC", "MLP"]
    # means = []
    # means_noise = []
    # for names in methods:
    #     scores, mean, std, total_points, mislabeled_points = test_of_classifier(coeff, labels, names, test_data_size=0.3)
    #     means.append(mean)
    #
    #
    # # Plot
    # methods_short = ["GNB", "SVM_poly", "SVM_linear", "KNN", "LDA", "DTC", "AdaBoost", "RandomForest", "GPC", "MLP"]
    # my_dpi = 100
    # plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    # # Set position of bar on X axis
    # barWidth = 0.33
    # br1 = np.arange(len(methods))
    # br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]
    #
    # plt.bar(br1, means, color='b', width=barWidth, align='center', label='without noise')
    # plt.xticks(rotation=45)
    # plt.xticks([r + barWidth - 0.33 / 2 for r in range(len(methods))], methods_short)
    # r = range(len(methods))
    # for i in range(len(means)):
    #     plt.text(r[i] - barWidth / 2, means[i] + 0.01, round(means[i], 2), color='b')
    # plt.ylabel('Accuracy')
    # plt.legend('lower right')
    # plt.savefig('Results/classification_on_simulated_data')
    # plt.show()


surface_properties_data_test()

#np_to_csv('')

#est.test_func()

#classification_test()
#np_to_csv('Results/Classification/classification_means_results.npy')
#np_to_csv('Results/Classification/classification_means_noise_results.npy')


#surface_properties_test()
#contact_centroid_estimation_test()
