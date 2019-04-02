from numpy.random import permutation
from sklearn import svm, datasets
import numpy as np
from sacred import Experiment

from sacred.observers import FileStorageObserver,MongoObserver


ex = Experiment('iris_rbf_svm')

# ex.observers.append(MongoObserver.create("mongodb+srv://user1:gY5fVQ1CwDL5SDjJ@cluster0-zhhvs.mongodb.net/test?retryWrites=true"))
# ex.observers.append(MongoObserver.create('mongodb://localhost:27017/'))
ex.observers.append(FileStorageObserver.create('my_runs3'))


def create_c():
    C=np.random.randint(0,5)
    return C



@ex.config
def cfg():
  C = create_c()
  gamma = 0.5



def essai_model(C,gamma):
    clf = svm.SVC(C, 'rbf', gamma=gamma)
    return clf


    list_C = [5, 0.2]
    list_gamma = [0.4, 0.6]


def vrai_main(C,gamma):
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]

    print(C,gamma)

    clf = essai_model(C, gamma)
    clf.fit(iris.data[:90], iris.target[:90])

    score = clf.score(iris.data[90:], iris.target[90:])

    ex.log_scalar("acc", score)

    pass



@ex.automain
def run(C, gamma):
    list_C = [5, 0.2]
    list_gamma = [0.4, 0.6]
    i=0
    for C in list_C:
        for gamma in list_gamma:
            print(i)

            vrai_main(C,gamma)

            i+=1

#Pas enregisté
