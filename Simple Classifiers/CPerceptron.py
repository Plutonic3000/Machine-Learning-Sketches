#!/usr/bin/python3
import _funx
class Perceptron(object):
    """ Classifier based on the perceptron

        Параметры
        ---------
        eta    : float - Темп обучения
        n_iter : int   - Проходы по трен. набору

        Атрибуты
        --------
        w_      : arr[] - Весовые коэф. после подгонки
        errors_ : list  - Число случаев ошиб. классификации в каждой эпохе
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        """ Выполнить подгонку модели под треню данные

            Параметры
            ---------
            X : массив, форма = [n_samples, n_features]
                тренировочные векторыб где
                n_samples  - число образцов
                n_features - чмсло признаков
            y : массив, форма = [n_samples] - Целевые значения

            Возвращает
            ----------
            self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        # возвращает массив нулей формы X на один элемент длиннее
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors      += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ Рассчитать чистый вход """
        return np.dot(X, self.w_[1:]) + self.w_[0];

    def predict(self, X):
        """ Вернуть метку класса после единичного скачка -
            (Выполняет функцию единичного скачка для чистого входа)"""
        return np.where(self.net_input(X) >= 0.0, 1, -1);

###############################

#plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#plt.xlabel('sepal length')
#plt.ylabel('petal length')
#plt.legend(loc='upper left')
#plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
#plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_, marker='o')
#plt.xlabel('Эпохи')
#plt.ylabel('Число случаев ошибочной классификации')
#plt.show()


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('длина чашелистника [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()
