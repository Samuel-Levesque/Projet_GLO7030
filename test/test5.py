from numpy.random import permutation
from sklearn import svm, datasets
import numpy as np
from sacred import Experiment

from sacred.observers import FileStorageObserver,MongoObserver


ex = Experiment('iris_rbf_svm')

# ex.observers.append(MongoObserver.create("mongodb+srv://user1:gY5fVQ1CwDL5SDjJ@cluster0-zhhvs.mongodb.net/test?retryWrites=true"))
# ex.observers.append(MongoObserver.create('mongodb://localhost:27017/'))
ex.observers.append(FileStorageObserver.create('my_runs4'))


def create_c():
    C=np.random.randint(1,5)
    return C



@ex.config
def cfg():
  C = create_c()
  gamma = 0.5



def essai_model(C,gamma):
    clf = svm.SVC(C, 'rbf', gamma=gamma)
    return clf




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
    ex.log_scalar("C",C)
    ex.log_scalar("gamma",gamma)

    pass



@ex.automain
def run(C, gamma):



    vrai_main(C,gamma)



#Pas enregisté
