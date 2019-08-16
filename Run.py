#Importing libraries and packages including out in-house class
import sys
sys.path.append(r'--Enter path of git repo---')
from TextClassifier import Classification_model
import pandas as pd
import win32com.client
xlapp = win32com.client.Dispatch("Excel.Application")
 
 
#Uploading labelled data to python, please ensure your data file has only two columns 1) Document's text and 2) Label
#In case there are multiple text columns that need to be used, please concatenate them in excel in one column
data_file = r"--Enter data file path---"
data = pd.read_excel(data_file)
data.columns = ["text", "label"]
 
 
#Creating copy of dataset
orig_df = data.copy()
 
 
#*********Initializing our text classification class and supplying only the text data to it (no labels)
EMC = Classification_model(Texts=orig_df['text'])
#*********Cleaning the data
CleanCorpus = EMC.CleanCorpus()
#*********Splitting into test and train samples with 30 percent test. This can be changed by updating value of test_percent. We also pass the labels to this function.
X_train, X_test, y_train, y_test =  EMC.train_test_split(y=orig_df['label'], test_percent=.3, return_required=True)
#*********Finally training of model is done in this step
classifier = EMC.train_model()
#*********Model testing in test set
y_pred_test = EMC.test_model_on_dev()
#*********Cross validation on whole dataset using 10 folds. This can be changed by updating num_folds parameter
EMC.cross_validation(num_folds=10)
#*********If you are  happy with model performance, then below code can be run to save the model
#**By default model will be saved to Classifier Pickles folder in base git repo directory
#**Ensure to updateAreyouSure to True before running code if you really want to save the model
EMC.save_model(AreYouSure=False)

#*********Prediction can be done using below function and passing it the new unlabeled text data as input
EMC_P = Classification_model(Texts=orig_df['text'], model_training=False)
CleanCorpus_P = EMC_P.CleanCorpus()
y_pred, y_pred_copy, Y_pred_prob, Y_pred_classed= EMC_P.predict_class()