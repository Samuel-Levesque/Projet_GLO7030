from numpy.random import permutation
from sklearn import svm, datasets

from sacred import Experiment

from sacred.observers import FileStorageObserver,MongoObserver


ex = Experiment('iris_rbf_svm')

# ex.observers.append(MongoObserver.create("mongodb+srv://user1:gY5fVQ1CwDL5SDjJ@cluster0-zhhvs.mongodb.net/test?retryWrites=true"))
ex.observers.append(MongoObserver.create('mongodb://localhost:27017/'))
# ex.observers.append(FileStorageObserver.create('my_runs'))


@ex.config
def cfg():
  C = 1.4
  gamma = 0.6

@ex.capture
def essae_function_caputre(C,gamma):
    e=C*gamma
    return e


@ex.automain
def run(C, gamma):
  iris = datasets.load_iris()
  per = permutation(iris.target.size)
  iris.data = iris.data[per]
  iris.target = iris.target[per]

  print("bonjour")
  essai=essae_function_caputre(C,gamma)
  print(essai)

  clf = svm.SVC(C, 'rbf', gamma=gamma)
  clf.fit(iris.data[:90],
          iris.target[:90])

  print("epoch numero 3")
  return clf.score(iris.data[90:],
                   iris.target[90:])