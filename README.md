# Classification_SVM
Classification of Breast cancer data using SVM(Support Vector Machine).
You can get this dataset from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) , I imported it from Scikit-learn using "from sklearn.datasets import load_breast_cancer ".

Cancer is classified in two categories: ['malignant' 'benign'] i.e referred under dict_keys(["target_names"]).
So I used Support vector classifier(SVC) from SVM to classify different cases into two categories malignant or benign.    