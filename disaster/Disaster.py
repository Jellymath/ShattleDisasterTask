import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # taken from https://history.nasa.gov/rogersrep/v1p146.htm, because origin file is total mess.
    temperature = [53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 80, 81]
    number_of_incidents = [3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    mapped = [1 if inc == 0 else 0 for inc in number_of_incidents]
    x = np.asarray(temperature).reshape(-1, 1)
    y = np.asarray(mapped)
    logit = LogisticRegressionCV().fit(x, y)
    gaussian = GaussianNB().fit(x, y)

    print("Accuracy for LogisticRegressionCV is {}".format(
        cross_val_score(LogisticRegressionCV(), x, y, cv=KFold(3, shuffle=True), scoring="accuracy").mean()))
    print("Accuracy for GaussianNB is {}".format(
        cross_val_score(GaussianNB(), x, y, cv=KFold(3, shuffle=True), scoring="accuracy").mean()))

    print("F1 Score for LogisticRegressionCV is {}".format(
        cross_val_score(LogisticRegressionCV(), x, y, cv=KFold(3, shuffle=True), scoring="f1").mean()))
    print("F1 Score for GaussianNB is {}".format(
        cross_val_score(GaussianNB(), x, y, cv=KFold(3, shuffle=True), scoring="f1").mean()))

    full_range = range(20, 90)
    print([(i, *logit.predict(i)) for i in full_range])
    print([(i, *gaussian.predict(i)) for i in full_range])


    def model(x):
        return 1 / (1 + np.exp(-x))

    print(logit.coef_)
    print(logit.intercept_)
    loss = [model(x * logit.coef_[0][0] + logit.intercept_[0]) for x in full_range]

    values = [(x, model(x * logit.coef_[0][0] + logit.intercept_[0])) for x in range(50, 110)]

    print(next(x for x in values if x[1] > 0.95))
    print(next(x for x in values if x[1] > 0.99))

    plt.plot(full_range, loss, color='red', linewidth=3)
    plt.axhline(y=0.95, color='y', linestyle='--')
    plt.scatter(full_range, [logit.predict(i) - 0.01 for i in full_range])
    plt.scatter(full_range, [gaussian.predict(i) + 0.01 for i in full_range])
    plt.scatter(temperature, mapped)
    plt.yticks([0, 1])
    plt.show()
