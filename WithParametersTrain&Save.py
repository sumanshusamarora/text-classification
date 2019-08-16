#########Importing required libraries
import sys
sys.path.append(r'C:\NotBackedUp\Common Repos\Text Classification')
from TextClassifier import Classification_model
import pandas as pd
import win32com.client
xlapp = win32com.client.Dispatch("Excel.Application")
##########Uploading training data
old_data_file = r"C:\NotBackedUp\Common Repos\Text Classification\Data\Old Training Data.xlsx"
new_data_file = r"C:\NotBackedUp\Common Repos\Text Classification\Data\New Labelled Data from user feedback.xlsx"
#old_data reading
data = pd.read_excel(old_data_file)
data.columns = ["text", "label"]
#new_data reading
data_object = xlapp.Workbooks.Open(new_data_file, False, True, None, "Data@2019", "Data@2019")
xlws = data_object.Sheets(1)
content = list(xlws.Range(xlws.Cells(1, 1), xlws.Cells(10000, 3)).Value)
xlapp.Workbooks.Close()
content = [row for row in content if row != (None, None, None)]
new_data = pd.DataFrame(columns=['subject', "body", "label"])
new_data['subject'] = [list(tup)[0] for tup  in content if list(tup)[0] != "Subject"]
new_data['body'] = [list(tup)[1] for tup  in content if list(tup)[1] != "Body"]
new_data['label'] = [list(tup)[2] for tup  in content if list(tup)[2] != "Label"]
new_data["text"] = new_data['subject'] + '\n' + new_data["body"]

#Mergine the two datasets
orig_df = data.copy()
orig_df = pd.concat((orig_df, new_data[["text", 'label']]), axis=0, ignore_index=True)

##########Training of data
EMC = Classification_model(Texts=orig_df['text'], model_training=True, RandomState=0)
Cleaned_Data = EMC.CleanCorpus()
X_train, X_test, y_train, y_test =  EMC.train_test_split(y=orig_df['label'], return_required=True, RandomOverSampling=True, test_percent=.3)
classifer = EMC.train_model(max_feat=850, inverse_regularisation=60)
y_pred, y_pred_prob, performance_metrics = EMC.test_model_on_dev()
avg_acc = EMC.cross_validation(num_folds=10)

#-------------------Run only if you would like to save the model, by default AreYouSure parameter is set to False
#EMC.save_model(AreYouSure=True)

#Prediction from model - Change orig_df['text'] with the new data
EMC_P = Classification_model(Texts=orig_df['text'], model_training=False)
Test_Corpus = EMC_P.CleanCorpus()
y_pred, y_pred_copy, y_pred_prob = EMC_P.predict_class()