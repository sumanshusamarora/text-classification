"Classification Model class is meant to use for text data classification training and prediction"
import CleaningText
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler
import TextClassifierExceptions
import joblib
import sys
import os
import heapq
import logging
import warnings


class Classification_model():  
    def __init__(self, Texts, model_training:bool = True, RandomState:int=100):
        #Initializing class variables which are accessible throughout the class definition
        self.Texts = Texts
        self.latestText = ""
        self.allTextStripped = ""
        #self.Text_series = None
        self.Text_list_latest = []
        self.Text_list_full = []
        self.model_training = model_training
        self.cleaned_corpus = None
        self.X = None
        self.y = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.temp_X_test = None
        self.temp_X_train = None
        self.temp_y_test = None
        self.temp_y_train = None
        self.classifier =  None
        self.Log_pipeline = None
        self.Log_pipeline_final = None
        self.performance_metrics = dict()
        self.RandomState = RandomState
        
        #For training purpose as we are going to pass series, list, array etc.
        try:
            self.Text_series = pd.Series(self.Texts)
        except TypeError("\033[0;31mText data not valid. Either 0 volume or wrong format provided as only list and str is allowed"):
            pass
            
        if len(self.Text_series) == 0:
            raise TypeError("\033[0;31mText data not valid. Either 0 volume or wrong format provided as only list and str is allowed") 
                   
    #Corpus cleaning func to clean corpus in case of Multi class model
    def CleanCorpus(self, scale:str='LS'):
        self.cleaned_corpus = None
        if scale not in ["LS","SS"]:
            raise ValueError("Only LS(Large Scale) or SS(Small Scale) are allowed as valid values for cleaning scale")
            
        if scale =='LS':
            disclaimer_removed = self.Text_series.apply(CleaningText.remove_disclaimer_from_body)
            self.cleaned_corpus  = CleaningText.create_clean_corpus(disclaimer_removed)
        else:
            self.cleaned_corpus  = CleaningText.clean_small_corpus(self.Text_series)
        return self.cleaned_corpus
                
    
    #Random Over Sampling since in case less and imbalanced data. If data is enough, we can set this off in train_test_split func
    def Random_Over_Sampling(self, X, y):
        if self.model_training == True:
            print("Executing Random Over Sampling")
            try:
                X = pd.Series(X)
                y = pd.Series(y)
            except: 
                raise ValueError("\033[0;31mSomething wrong with X or y values. Please check.")
                
            try:  
                ROS = RandomOverSampler(random_state=self.RandomState)
                data_array = ROS.fit_resample(pd.DataFrame(X), pd.DataFrame(y))
                data_new = pd.DataFrame(list(zip(data_array[0], data_array[1])))
                data_new.columns = ['text', 'label']
                data_new['text'] = data_new['text'].apply(lambda x: x[0])
                X_new = list(data_new.iloc[:,0])
                y_new = data_new.iloc[:,1]
            except:
                raise TypeError("\033[0;31mSomething is wrong with Random Sampling. Please check.")
            return X_new, y_new
        
    #Train test split function only runs while training and can also call Random OverSampling if data is less. Default split value is
    #30% but can be changed by passing the parameter to function. Return can be requested by passing True paramater for return required
    def train_test_split(self, y, test_percent=0.3, return_required:bool = False, RandomOverSampling:bool = True):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        if self.model_training == True:
            try:
                y = pd.Series(y)
            except:
                raise ValueError("\033[0;31mSomething wrong with X or y values. Please check.")
            
            try:
                if self.cleaned_corpus == None:
                    raise TextClassifierExceptions.DataNotCorrect
            except:
                raise ValueError("\033[0;31mCorpus cleaning is mandatory before train test splitting")
                  
                
            if RandomOverSampling == True:
                print("\033[0;30mExecuting Train Test Split with Oversampling")
                self.X, self.y = self.Random_Over_Sampling(self.cleaned_corpus, y)
                
            if RandomOverSampling == False:
                print("\033[0;30mExecuting Train Test Split without Oversampling")
                self.X = self.cleaned_corpus
                self.y = y
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_percent, random_state = self.RandomState)
            if return_required == True:
                return self.X_train, self.X_test, self.y_train, self.y_test
        
        
    #Training function, runs only when model_training is set to True during class initialization. You can use different training data
    #as well in case you already have cleaned data and do not want to use the class function by setting Override_existing_X_y to True and 
    #passing your X_trai n & y_train parameters. It returns a pipeline which us used by test model fucntion
    def train_model(self, X_train=None, y_train=None, Override_existing_X_y:bool = False, max_feat:int=900, inverse_regularisation:int = 70, Tf_Idf:bool=True):
        if self.model_training == True:
            try:
                if self.X_train == None:
                    raise TextClassifierExceptions.IncorrectOverrideError
            except:
                raise TypeError("\033[0;31mTrain_test_split did not run correctly, please fix that first")                
                
            self.Log_pipeline = None
            self.temp_X_train = None
            self.temp_y_train = None
             
            try:
                if Override_existing_X_y == True:
                    if X_train==None or y_train==None:
                        raise TextClassifierExceptions.IncorrectOverrideError
                    
                    if Override_existing_X_y == False:
                        if X_train!=None or y_train!=None:
                            raise TextClassifierExceptions.IncorrectOverrideError
                            
            except:
                raise ValueError("\033[0;31mOverride value does not concide with X & y inputs.\033[0;31m Execution stopped")
                    
                    
            if X_train==None or y_train==None:
                self.temp_X_train = self.X_train
                self.temp_y_train = self.y_train
            else:
                self.temp_X_train = X_train
                self.temp_y_train = y_train
                
            print("Creating pipeline")
            if Tf_Idf == False:
                Log_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('clf', LogisticRegression(C=inverse_regularisation, dual=False, multi_class='ovr', penalty='l1', solver='saga', tol=0.001, max_iter=1500, random_state=self.RandomState))
                       ])
            else:
                Log_pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), max_features=max_feat, stop_words='english')),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression(C=inverse_regularisation, dual=False, multi_class='ovr', penalty='l1', solver='saga', tol=0.001, max_iter=1500, random_state=self.RandomState))
                       ])
            self.Log_pipeline = Log_pipeline.fit(self.temp_X_train, self.temp_y_train)
            
            try:
                if self.Log_pipeline == None:
                    raise TextClassifierExceptions.IncorrectOverrideError
                    self.temp_X_train = None
                    self.temp_y_train = None
                
            except:
                raise TypeError("\033[0;31mPipeline not created successfully, please check your parameters")
                
            return self.Log_pipeline
        
    #Test function, runs only when model_training is set to True during class initialization. You can use different testing data
    #as well if you like by setting Override_existing_X_y to True and passing your X_test & y_test parameters. 
    def test_model_on_dev(self, X_test=None, y_test=None, Override_existing_X_y:bool = False, metrics = [accuracy_score, f1_score, confusion_matrix, classification_report]):
        if self.model_training == True:
            try:
                if self.Log_pipeline == None:
                    raise TextClassifierExceptions.IncorrectOverrideError
                
            except:
                raise TypeError("\033[0;31mTrain model step did not return the pipeline correctly")
                
            self.performance_metrics = dict()
            self.temp_X_test = None
            self.temp_y_test = None
            
            try:
                if Override_existing_X_y == True:
                    if X_test==None or y_test==None:
                        raise TextClassifierExceptions.IncorrectOverrideError
                        
                if Override_existing_X_y == False:
                    if X_test!=None or y_test!=None:
                        raise TextClassifierExceptions.IncorrectOverrideError
            except:
                raise ValueError("\033[0;31mOverride value does not concide with X & y inputs.\033[0;31m Execution stopped")
                
                    
            if X_test==None or y_test==None:
                self.temp_X_test = self.X_test
                self.temp_y_test = self.y_test
            else:
                self.temp_X_test = X_test
                self.temp_y_test = y_test
                    
            y_pred = self.Log_pipeline.predict(self.temp_X_test)
            y_pred = pd.Series(y_pred)
            
            y_pred_prob = self.Log_pipeline.predict_proba(self.temp_X_test)
            
            
            for metric in metrics:
                m_name = str(metric).split(" ")[1].title()
                try:
                    print('\033[0;30m '+m_name + ' results ------> \n')
                    print(metric(y_pred, self.temp_y_test))
                    if metric in [accuracy_score, f1_score]:
                        self.performance_metrics[m_name] = metric(y_pred, self.temp_y_test)
                except:
                    print("\033[0;31m '"+ m_name + "' \033[0;31m not valid for this data")
                print('---------------------------------------------------- \n \n')
            return y_pred, y_pred_prob, self.performance_metrics
        
        
    ##In case a cross validation needs to be run
    def cross_validation(self, num_folds:int=10):
        if self.model_training == False:
            raise ValueError("\033[0;34mCross validation not allowed when model_training is set to False. Review your class initialzation options")
    
        else:
            if self.X == None:
                raise ValueError("\033[0;34mRunning train test split is mandatory before running cross validation")
            else:
                print("Running cross validation on {} folds".format(num_folds))
                LCV = cross_validate(self.Log_pipeline, self.X, self.y, cv=num_folds, scoring=['accuracy'])
                avg_acc = np.mean(LCV['test_accuracy'])
                print("Average accuracy after running {} folds is {}".format(num_folds, str("{0:.2f}".format(avg_acc))))
                return avg_acc


    #Save model function runs only when model_training is set to True during class initialization. AreYouSure parameter is by default 
    #set to False, please update to True if save is required. Model name and folder location can also be changed
    def save_model(self, AreYouSure:bool=False, ModelLocation:str='', ModelName:str = 'ClassifierMultiRetrained', train_on_test_set = True):
        if self.model_training == True:
            if AreYouSure == True:
                try:
                    if self.Log_pipeline == None:
                        raise TextClassifierExceptions.DataNotCorrect
                except:
                    raise TypeError("\033[0;31mPipeline not created, make sure training process was completed")
                    
                if train_on_test_set == True:
                    self.Log_pipeline_final = self.Log_pipeline.fit(self.temp_X_test, self.temp_y_test)
                else:
                    self.Log_pipeline_final = self.Log_pipeline
                    
                if ModelLocation == '':
                    ModelLocation = sys.path[len(sys.path)-1] + "\\Classifier Pickles"
                try:
                    model = os.path.join(ModelLocation, ModelName+'.pkl')
                    joblib.dump(self.Log_pipeline_final, model, compress=3)
                    print("\033[0'34mModel successfully saved at {}".format(model))
                except:
                    raise TypeError("Model location incorrect, please provide location as parameter to function")
                return self.Log_pipeline_final
            else:
                print("\033[0;34mSeems like you are not sure about saving the new model. Please review your options")
        else:
            print("\033[0;34mModel saving not allowed when model_training is set to False. Review your class initialzation options")
            
        
    #Multiclass Predict model class for prediction purpose. The prod data (one Text in dict format) and bulk predictions during training 
    #can be done through this func. In case you models is stored somehwere else, please pass folder location and model name as well
    def predict_class(self, model_location:str = '', model_name:str = 'ClassifierMultiRetrained.pkl', top_two_diff_uncat = 0.5):
        y_pred = None
        Y_pred_prob = None
        
        if self.model_training == True:
            raise ValueError("\033[0;31mPrediction not allowed when model_training is set to True. Review your class initialzation options")
        
        if self.model_training == False:
            
            if self.cleaned_corpus == None:
                raise TypeError("\033[0;31m Corpus cleaning is mandatory before train test splitting")
            
            if self.cleaned_corpus != None:
                if model_location == '':
                    model_location = str(sys.path[len(sys.path)-1]) + "\\Classifier Pickles"
                print("\033[0;30mLoading the saved model from location - " + str(model_location))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    full_model_path = os.path.join(model_location, model_name)
                    print("Full model path - " + full_model_path)
                    model = joblib.load(full_model_path)
                print("\033[0;30mModel loaded successfully, getting ready for prediction")
                y_pred = pd.Series(model.predict(self.cleaned_corpus))
                Y_pred_prob = model.predict_proba(self.cleaned_corpus)
                Y_pred_prob.copy()
                y_pred_copy = y_pred.copy()
                
                print("\033[0;30mPredictions completed, labelling uncategorised Texts")
                for i in range(Y_pred_prob.shape[0]):
                    max_value_each_line = heapq.nlargest(2, Y_pred_prob[i])
                    if max(max_value_each_line) - min(max_value_each_line) < top_two_diff_uncat:
                        y_pred_copy[i] = "Uncategorised"
                print("\033[0;30mPredicted data ready for review")
                 #Creating log in case of prod prediction
                #if self.Message_dict != None:
                #    if y_pred_copy[0] != "Uncategorised":
                #        logging.debug('Multi Class model classified message [{}] as :[{}] with prob [{}]'.format(self.cleaned_corpus[0], y_pred_copy[0], str(max(Y_pred_prob[0]))))
                #    else:
                #        logging.debug('Multi Class model classified message [{}] as :[{}] because there was no clear win for any category'.format(self.cleaned_corpus[0], y_pred_copy[0]))
                        
                return y_pred, y_pred_copy, Y_pred_prob, model.classes_
            