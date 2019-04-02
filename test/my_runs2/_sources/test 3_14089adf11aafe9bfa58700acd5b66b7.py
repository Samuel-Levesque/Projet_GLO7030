from numpy.random import permutation
from sklearn import svm, datasets

from sacred import Experiment

from sacred.observers import FileStorageObserver,MongoObserver


ex = Experiment('iris_rbf_svm')

# ex.observers.append(MongoObserver.create("mongodb+srv://user1:gY5fVQ1CwDL5SDjJ@cluster0-zhhvs.mongodb.net/test?retryWrites=true"))
# ex.observers.append(MongoObserver.create('mongodb://localhost:27017/'))
ex.observers.append(FileStorageObserver.create('my_runs2'))


@ex.config
def cfg():
  list_C = [1.4,1.7]
  list_gamma = [0.4,0.6]



def essai_model(C,gamma):
    clf = svm.SVC(C, 'rbf', gamma=gamma)
    return clf



@ex.automain
def run(list_C, list_gamma):
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]

    print("bonjour")

    for C, gamma in zip(list_C,list_gamma):

        print(C)
        print(gamma)

        clf = svm.SVC(C, 'rbf', gamma=gamma)
        clf.fit(iris.data[:90],
              iris.target[:90])

        print("epoch numero 3")


    return clf.score(iris.data[90:],
                   iris.target[90:])


