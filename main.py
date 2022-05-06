import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


data = pd.read_csv("covid.data")
print('Shape:', data.shape)
print(data)
print("\nAverages:\n", data.describe())
print(data.info())

data['Breathing Problem']=le.fit_transform(data['Breathing Problem'])
data['Fever']=le.fit_transform(data['Fever'])
data['Dry Cough']=le.fit_transform(data['Dry Cough'])
data['Sore throat']=le.fit_transform(data['Sore throat'])
data['Running Nose']=le.fit_transform(data['Running Nose'])
data['Asthma']=le.fit_transform(data['Asthma'])
data['Chronic Lung Disease']=le.fit_transform(data['Chronic Lung Disease'])
data['Headache']=le.fit_transform(data['Headache'])
data['Heart Disease']=le.fit_transform(data['Heart Disease'])
data['Diabetes']=le.fit_transform(data['Diabetes'])
data['Hyper Tension']=le.fit_transform(data['Hyper Tension'])
data['Fatigue ']=le.fit_transform(data['Fatigue '])
data['Gastrointestinal ']=le.fit_transform(data['Gastrointestinal '])
data['Abroad travel']=le.fit_transform(data['Abroad travel'])
data['Contact with COVID Patient']=le.fit_transform(data['Contact with COVID Patient'])
data['Attended Large Gathering']=le.fit_transform(data['Attended Large Gathering'])
data['Visited Public Exposed Places']=le.fit_transform(data['Visited Public Exposed Places'])
data['Family working in Public Exposed Places']=le.fit_transform(data['Family working in Public Exposed Places'])
data['Wearing Masks']=le.fit_transform(data['Wearing Masks'])
data['Sanitization from Market']=le.fit_transform(data['Sanitization from Market'])
data['COVID-19']=le.fit_transform(data['COVID-19'])
print(data)

x = data.drop('COVID-19', axis=1)
y = data['COVID-19']

#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#creating the model
model = LogisticRegression()
#fiting the model
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#accuracy multiplied to give percentage
accuracy_lr = model.score(x_test, y_test) * 100
print(accuracy_lr)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(model, x_test, y_test)
plt.show()



#creating the model
RFC = RandomForestClassifier(n_estimators=2000)
#fiting the model
RFC.fit(x_train, y_train)
y_pred = RFC.predict(x_test)
#accuracy multiplied to give percentage
accuracy_rf = RFC.score(x_test, y_test) * 100
print(accuracy_rf)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(RFC, x_test, y_test)
plt.show()

#creating the model
knn = KNeighborsClassifier(n_neighbors=20)
#fiting the model
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
#accuracy multiplied to give percentage
accuracy_knn = knn.score(x_test, y_test) * 100
print(accuracy_knn)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(knn, x_test, y_test)
plt.show()


#creating the model
GNB = GaussianNB()
#fiting the model
GNB.fit(x_train,y_train)
y_pred = GNB.predict(x_test)
#accuracy multiplied to give percentage
accuracy_gNB = GNB.score(x_test, y_test)*100
print(accuracy_gNB)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(GNB, x_test, y_test)
plt.show()

#creating the model
svm = svm.SVC(kernel='linear')
#fiting the model
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
#accuracy multiplied to give percentage
accuracy_svm=svm.score(x_test, y_test)*100
print(accuracy_svm)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(svm, x_test, y_test)
plt.show()







