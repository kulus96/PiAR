from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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

    def split_data(self,data_x,data_y,test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.5, random_state=0)

    def fit_data(self):
        if self.method_name == "GNB":
            self.model.fit(self.X_train, self.y_train)

    def predict(self,data):
        if self.method_name == "GNB":
            return self.model.predict(data)


## Running Gaussian Naive Baysian
X, y = load_iris(return_X_y=True)
C1 = Classifier("GNB")

C1.split_data(X,y,0.8)
C1.fit_data()
y_pred = C1.predict(C1.X_test)

print("Number of mislabeled points out of a total %d points : %d" % (C1.X_test.shape[0], (C1.y_test != y_pred).sum()))