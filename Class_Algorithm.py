import pandas as pd


# To use Breast cancer dataset we have to import it from Scikit learn
from sklearn.datasets import load_breast_cancer

data_cancer = load_breast_cancer()

#To check the main keys of the data set 
print(data_cancer.keys())
# It will print out the dictionary of all the keys.

#To check the target value or its name
print(data_cancer['target'],'\n',data_cancer['target_names'])


#Build a dataframe of features and data about it using pandas library:
df_dataFeatures = pd.DataFrame(data_cancer['data'], columns = data_cancer['feature_names'])
# To check the head of the dataframe:(You can use a jupyter library or you can append it to a .CSV file, if it is not working in the Sublime text)
print(df_dataFeatures.head())


#Now, one of the most important Step to split your data in two parts i.e 1) Traning Set, 2) Test Set
#To do the above we have to import train_test_split method from Scikit-learn

from sklearn.model_selection import train_test_split 
X = df_dataFeatures
y = data_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 
#Here I have kept the test data size to be 30% and left is used to train the Machine 

#Now, a really important part to Train the Machine.
#Here I am using SVM(Support Vector Machine) for Classification purpose.

#We will use SVC class from SVM in scikit-learn
from sklearn.svm import SVC

#Train your machine i.e your Support Vector Classifier(SVC)
#Train/Fit your model with training data
SVC().fit(X_train, y_train)


#You will get a warning message if your python version is using the old default values for gamma or C''. 

#Now predict using the default values to check whether it works perfectly or not:

prediction = SVC().fit(X_train, y_train).predict(X_test)


