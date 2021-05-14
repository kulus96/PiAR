from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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

        if method_name == "GNB":
            self.model = GaussianNB()
        elif method_name == "SVM_poly":
            self.model = svm.SVC(kernel='poly', degree=9)
        elif method_name == "SVM_linear":
            self.model = svm.SVC(kernel='linear')
        elif method_name == "Bagging":
            self.model = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=10, random_state=0)
        elif method_name == "k-nearest neighbors":
            self.model = KNeighborsClassifier(n_neighbors=3)    # Lower n_neighbors is better
        elif method_name == "LDA":
            self.model = LinearDiscriminantAnalysis(solver='lsqr')
        elif method_name == "DTC":
            self.model = DecisionTreeClassifier(random_state=0)
        elif method_name == "AdaBoostClassifier":
            self.model = AdaBoostClassifier(n_estimators=50, random_state=False, learning_rate=0.5)
        elif method_name == "RandomForest":
            self.model = RandomForestClassifier()
        elif method_name == "GPC":
            self.model = GaussianProcessClassifier(max_iter_predict=100)
        elif method_name == "MLP":
            self.model = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.01, max_iter=10000)


    def split_data(self,data_x,data_y,test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=0)

    def fit_data(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, data):
        return self.model.predict(data)

def test_of_classifier(X,y,method_name,test_data_size):

    model = Classifier(method_name)
    model.split_data(X, y, test_data_size)
    model.fit_data()
    y_pred = model.predict(model.X_test)

    print(model.method_name,
          " : Number of mislabeled points out of a total %d points : %d" % (model.X_test.shape[0],
                                                                            (model.y_test != y_pred).sum()))
    scores = cross_val_score(model.model, model.X_test, model.y_test, cv=2)
    #print(scores)
    print(model.method_name,
          " : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    return scores, scores.mean(), scores.std(), model.X_test.shape[0], (model.y_test != y_pred).sum()

    #confusion_mat = confusion_matrix(model.y_test, y_pred)
    #report = classification_report(model.y_test, y_pred)

    #print(confusion_mat)
    #print(report)
    print()


'''
## Running Gaussian Naive Baysian
X = load_csv("surface_properties.csv")
y = load_csv("label.csv")

#X = load_csv("C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/coefficients.csv")
#y = load_csv("C:/Users/Andreas/Dropbox/8. semester/Project in Advanced Robotics/Git/PiAR/Python_estimation/labels.csv")



test_data_size = 0.5

for i in range(len(X)):
    X[i][0] += np.random.normal(0, 0.05)
    X[i][1] += np.random.normal(0, 0.05)


#methods = ["GNB", "SVM_poly", "SVM_linear", "Bagging", "k-nearest neighbors", "LDA", "DTC"]
methods = ["GNB", "SVM_poly", "SVM_linear", "k-nearest neighbors", "LDA", "DTC", "AdaBoostClassifier", "RandomForest", "GPC", "MLP"]

for names in methods:
    test_of_classifier(X, y, names,test_data_size)

'''




"""
C_GNB = Classifier("GNB")
C_GNB.split_data(X,y,test_data_size)
C_GNB.fit_data()
y_pred = C_GNB.predict(C_GNB.X_test)

print(C_GNB.method_name," : Number of mislabeled points out of a total %d points : %d" % (C_GNB.X_test.shape[0],
                                                                                          (C_GNB.y_test != y_pred).sum()))
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
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

"""




