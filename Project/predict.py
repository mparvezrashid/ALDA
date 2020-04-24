import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


class SVM():
    def __init__(self):
        self.classifier=SVC()
        pass


    def classify(self, X_train, y_train, X_val, X_test,):
        self.classifier = SVC(C=1, kernel='linear', probability=True,
                         class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        y_test_predict = self.classifier.predict(X_test)
        y_val_predict = self.classifier.predict(X_val)

        return y_test_predict,y_val_predict

class RF():
    def __init__(self):
        self.classifier=RandomForestClassifier()
        pass

    def classify(self, X_train, y_train, X_val, X_test,):
        self.classifier = RandomForestClassifier(n_estimators=500,class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        y_test_predict = self.classifier.predict(X_test)
        y_val_predict = self.classifier.predict(X_val)

        return y_test_predict,y_val_predict

class LR():
    def __init__(self):
        self.classifier=RandomForestClassifier()
        pass

    def classify(self, X_train, y_train, X_val, X_test,):
        self.classifier = RandomForestClassifier(n_estimators=500,class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        y_test_predict = self.classifier.predict(X_test)
        y_val_predict = self.classifier.predict(X_val)

        return y_test_predict,y_val_predict


class predict(object):

    def __init__(self, N, X_all,y_all,X_test,y_test,classifier):
        self.initial_lab_instance = N
        self.X_all = X_all
        self.y_all = y_all
        self.X_test = X_test
        self.y_test = y_test
        self.qry=N
        self.classifier = classifier


    def select_N_instance(self, N, X_all, y_all):
        #rand = check_random_state(0)
        #print(len(X_all))
        rand_index = np.random.choice(len(X_all), N, replace=False)
        #print(rand_index)
        X_train_new = X_all[rand_index]


        y_train_new = y_all[rand_index]
        return rand_index, X_train_new,y_train_new


    def uncertain_sample(self, X_val_prob, N):
        Entropy=(- X_val_prob * np.log2(X_val_prob)).sum(axis=1)
        return (np.argsort(Entropy)[::-1])[:N]

    def random_sample(self,X_val_prob, N):
        rand = check_random_state(0)
        return np.random.choice(X_val_prob.shape[0],N,replace=False)




    def get_accuracy(self, y_predict, y_actual):
        accuracy = np.mean(y_predict.ravel() == y_actual.ravel())
        return accuracy
    def Train(self):
        index, X_train, y_train = self.select_N_instance(self.initial_lab_instance, self.X_all, self.y_all)
        accuracy=[]
        X_val = np.copy(self.X_all)
        X_val = np.delete(X_val,index,axis=0)
        y_val = np.copy(self.y_all)
        y_val = np.delete(y_val,index,axis=0)

        if self.classifier == 'SVM':
            model = SVM()
        elif self.classifier == 'RF':
            model = RF()

        y_test_predict, y_val_predict = model.classify(X_train,y_train,X_val,self.X_test)
        #print(self.get_accuracy(y_test_predict, self.y_test))
        #print("Test Accuracy: " + str(self.get_accuracy(y_test_predict, self.y_test)))
        accuracy.append(self.get_accuracy(y_test_predict, self.y_test))
        query_limit = 100
        while self.qry < query_limit:
            X_val_prob = model.classifier.predict_proba(X_val)
            uncertain_instance = self.uncertain_sample(X_val_prob,self.initial_lab_instance)
            #uncertain_instance = self.random_sample(X_val_prob,self.initial_lab_instance)
            #print(uncertain_instance)
            X_train = np.concatenate((X_train, X_val[uncertain_instance]))
            y_train = np.concatenate((y_train, y_val[uncertain_instance]))
            X_val = np.delete(X_val, uncertain_instance, axis=0)
            y_val = np.delete(y_val, uncertain_instance, axis=0)
            self.qry+=self.initial_lab_instance
            y_test_predict, y_val_predict = model.classify(X_train, y_train, X_val, self.X_test)
            #print("Test Accuracy: "+str(self.get_accuracy(y_test_predict, self.y_test)))
            accuracy.append(self.get_accuracy(y_test_predict, self.y_test))
            return(np.mean(accuracy))

    def LRTrain(self):

        if self.classifier == 'LR':
            #print("Baal")
            model2 = LogisticRegression(random_state=0).fit(self.X_all, self.y_all)
            # implement the logistic model
            #y_pred = model.predict(self.X_test)
            scr = model2.score(self.X_test, self.y_test)
            #model.predict_proba(x_test)
            print(scr)
            return scr




