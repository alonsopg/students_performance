import pandas as panda
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_decomposition

#Se crea un vectorizador
dict_vectorizer = DictVectorizer()
#Se carga la información el archivo donde se encuentran los datos
training_data = panda.read_csv('student-mat.csv')
#Se crea un diccionario de los datos
training_data_dict = training_data.T.to_dict().values()
#Se genera una matriz a partir de los datos cargados
training_matrix = dict_vectorizer.fit_transform(training_data_dict)





print training_data_dict


#Vectorizing

print training_matrix.toarray()


labels = training_data['sex']
testing_data = panda.read_csv('student-mat_test.csv')
testing_data_dict = testing_data.T.to_dict().values()
print testing_data_dict


testing_matrix = dict_vectorizer.transform(testing_data_dict)
print testing_matrix.toarray()


from sklearn.svm import SVC

#C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001,
#  cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None
#ESTA ES LA LINEA QUE SE SUSTITUYO
#svm = SVC(kernel='rbf', class_weight='auto')
#AQUI ESTA LA PAGINA DONDE SE PUEDEN COSULTAR LOS POSIBLES VALORES PARA CADA HIPERPARAMETRO
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
svm=SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,
        tol=0.001, cache_size=200, class_weight="auto", verbose=False, max_iter=-1, random_state=None)
svm.fit(training_matrix, labels)
result = svm.predict(testing_matrix)
print result

c=svm.score(training_matrix, labels)
print "este es el score de precision con 10 features etiquetados:\n", c
