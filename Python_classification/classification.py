from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for index_row, row in enumerate(csv_reader):
            if len(row) == 1:
                data.append(int(row[0]))
            else:
                data_row = []
                for index_elements,elements in enumerate(row):
                    data_row.append(float(elements))
                data.append(data_row)

    return np.array(data)

# A Sample class with init method
class Classifier:

    def __init__(self, method_name):
        self.method_name = method_name

        self.X_train = []
        self.X_test = []

        self.y_train = []
        self.y_test = []

        if method_name == "GNB" :
            self.model = GaussianNB()
        elif method_name == "SVM_poly":
            self.model = svm.SVC(kernel='poly')
        elif method_name == "SVM_linear":
            self.model = svm.SVC(kernel='linear')
        elif method_name == "Bagging":
            self.model = BaggingClassifier(base_estimator=svm.SVC(),n_estimators=10, random_state=0)

    def split_data(self,data_x,data_y,test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=0)

    def fit_data(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self,data):
        return self.model.predict(data)


## Running Gaussian Naive Baysian
X = load_csv("surface_properties.csv")
y = load_csv("label.csv")

test_data_size = 0.3

C_GNB = Classifier("GNB")
C_GNB.split_data(X,y,test_data_size)
C_GNB.fit_data()
y_pred = C_GNB.predict(C_GNB.X_test)

print(C_GNB.method_name," : Number of mislabeled points out of a total %d points : %d" % (C_GNB.X_test.shape[0],
                                                                                          (C_GNB.y_test != y_pred).sum()))
C_SVM_poly = Classifier("SVM_poly")
C_SVM_poly.split_data(X,y,test_data_size)
C_SVM_poly.fit_data()
y_pred = C_SVM_poly.predict(C_SVM_poly.X_test)

print(C_SVM_poly.method_name," : Number of mislabeled points out of a total %d points : %d" % (C_SVM_poly.X_test.shape[0],
                                                                                          (C_SVM_poly.y_test != y_pred).sum()))

C_SVM_linear = Classifier("SVM_linear")
C_SVM_linear.split_data(X,y,test_data_size)
C_SVM_linear.fit_data()
y_pred = C_SVM_linear.predict(C_SVM_linear.X_test)

print(C_SVM_linear.method_name," : Number of mislabeled points out of a total %d points : %d" % (C_SVM_linear.X_test.shape[0],
                                                                                          (C_SVM_linear.y_test != y_pred).sum()))

C_SVM_bagging = Classifier("Bagging")
C_SVM_bagging.split_data(X,y,test_data_size)
C_SVM_bagging.fit_data()
y_pred = C_SVM_bagging.predict(C_SVM_bagging.X_test)

print(C_SVM_bagging.method_name," : Number of mislabeled points out of a total %d points : %d" % (C_SVM_bagging.X_test.shape[0],
                                                                                          (C_SVM_bagging.y_test != y_pred).sum()))


data_plot = X
static_fric = [item[0] for item in data_plot]
dynamic_fric = [item[1] for item in data_plot]
fig, ax = plt.subplots()
plt.scatter(static_fric, dynamic_fric, c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("Plot of data")
ax.set_xlabel("static friction")
ax.set_ylabel("dynamic friction")
plt.show()






