import pandas as pd
import numpy as np
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
#print(df_dataFeatures.head())


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

#<<<<<<<<<<<<<<< ADDING PIPELINE TO THE MODEL >>>>>>>>>>>>>>>
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


pipeline = Pipeline([('scaler', StandardScaler()), ('SVM', SVC())])

pipeline.fit(X_train,y_train)

Predictions = pipeline.predict(X_test)
#Train your machine i.e your Support Vector Classifier(SVC)
#Train/Fit your model with training data
#SVC().fit(X_train, y_train)


#You will get a warning message if your python version is using the old default values for gamma or C''. 

#Now predict using the default values to check whether it works perfectly or not:

#prediction = SVC().fit(X_train, y_train).predict(X_test)


#Now check the classification report and the confusion metrix
#To do the above we first have to import the classification_report & confision_matrix from "metrics" class of scikit-learn

from sklearn.metrics import confusion_matrix, classification_report  

#print out the final results

#print(confusion_matrix(y_test, rediction),'\n', classification_report(y_test, prediction))

""" I got UNDESIRED results for classification. 
    As I got 0.00 predicted samples for '0' Targert as the classification result.
     <<<< REFER TO Screenshot Capture3 in project_screenshot folder"""

#To overcome this issue I can try with the different values for gamma or C

# WE CAN USE GRID SEARCH from Scikit-learn.
"""AS LARGER 'C' VALUE CAN RESULT IN LOW BIAS AND HIGH VARIANCE
HIGH GAMMA CAN RESULT IN VERY LOW VARIANCE"""
#from sklearn.model_selection import GridSearchCV 

#Test out different manual values for C and Gamma
# And check for the best pair of gamma and C
# param_grid = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}
# Grid = GridSearchCV(SVC(), param_grid, verbose = 4 )	

#Verbose is the text output of the description of the process, so assign it some value otherwise it will not display what the process is doing
#In the above SVC() is the estimator, param_grid are the mannual values for gamma and C

#Grid.fit(X_train, y_train)

#print(Grid.best_params_)
# As a result I got the best parameters as C=10 and gamma = 0.0001


# Make a new prediction using the custom(BEST FOUND) values for C and gamma
#prediction2 = SVC(C = 10, gamma =0.0001).fit(X_train, y_train).predict(X_test)
c_m = confusion_matrix(y_test, Predictions)
c_r = classification_report(y_test, Predictions)

print(c_m,'\n', c_r)

# AS the final result I got a CLASSIFICATION of 0.95 to 0.97 in both 0 and 1 <<<REFER to Screenshot Capture5 in project_screenshots folder>>>

  #Now Lets add some graphs or plot for the Confusion_matrix 

import matplotlib.pyplot as plt 
#%matplotlib inline
    #<<%Matplotlib will not work in normal Sublime environment
plt.imshow(c_m, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('SVM Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
	for j in range(2):
		plt.text(i,j, str(s[i][j])+'='+str(c_m[i][j]))
plt.show()		
                    #<<<TO CHECK THE RESULTS OF PLOT SEE Capture6 in project_screenshots folder>>>

####Run the Code in jupyter notebook 