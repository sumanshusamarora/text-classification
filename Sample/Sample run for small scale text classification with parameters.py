#########Importing required libraries
import sys
sys.path.append(r'C:\Users\suman\text-classification-master')
from TextClassifier import Classification_model
import pandas as pd
##########Uploading training data
data_path = r"F:\Work\Upwork\Music Text Entity Matching\MasterEntityCategorization.xlsx"
data = pd.read_excel(data_path)
data["label"] = data["Category 1"] + " " + data["Category 2"]
data.dropna(inplace=True)
#Mergine the two datasets
orig_df = data.reset_index(drop=True)

orig_df["label"].value_counts().plot(kind='bar')

##########Training of data
EMC = Classification_model(Texts=orig_df['Entity Name'], model_training=True, RandomState=0)
Cleaned_Data = EMC.CleanCorpus(scale='SS')
#X, Y = EMC.Random_Over_Sampling(orig_df['Entity Name'], orig_df['label'])
X_train, X_test, y_train, y_test =  EMC.train_test_split(y=orig_df['label'], return_required=True, RandomOverSampling=True, test_percent=.3)
classifer = EMC.train_model(max_feat=850, inverse_regularisation=30, Tf_Idf=False)
y_pred, y_pred_prob, performance_metrics = EMC.test_model_on_dev()
avg_acc = EMC.cross_validation(num_folds=10)
final_classifer = EMC.save_model(AreYouSure=True, IncludeTestData=True)


#Prediction from model - Change orig_df['text'] with the new data
data_pred_path = r"F:\Work\Upwork\Music Text Entity Matching\labels.xlsx"
data_pred = pd.read_excel(data_pred_path)
data_pred.dropna(inplace=True)
data_pred.reset_index(drop=True, inplace=True)
data_pred_copy = data_pred.copy()
data_pred_copy['label'] = data_pred_copy['label'].apply(lambda x: x.replace("WM", "Warner Music"))


EMC_P = Classification_model(Texts=data_pred_copy['label'], model_training=False, RandomState=0)
cleaned_test_data = EMC_P.CleanCorpus(scale='SS')
y_pred, y_pred_copy, Y_pred_prob, y_pred_prob_classes_ = EMC_P.predict_class(top_two_diff_uncat=0.4)

Y_pred_prob = pd.DataFrame(Y_pred_prob)
Y_pred_prob.columns = list(y_pred_prob_classes_)

y_pred_copy_df = pd.DataFrame(y_pred_copy)

def divide_in_two_cats(x):
    x_split = x.split()
    cat1 = x_split[0]
    try:
        cat2 = x_split[1]
    except:
        cat2 = ""
        pass
    return [cat1, cat2]

y_pred_copy_df = pd.DataFrame(y_pred_copy.apply(divide_in_two_cats))
y_pred_copy_df["Category 1"] = ""
y_pred_copy_df["Category 2"] = ""
for i in range(len(y_pred_copy_df)):
    y_pred_copy_df.at[i,"Category 1"] = y_pred_copy_df[0][i][0]
    y_pred_copy_df.at[i,"Category 2"] = y_pred_copy_df[0][i][1]

y_pred_copy_df = y_pred_copy_df.drop(columns=0)

#Delibrately keeping data_pred in-stead-of data_pred_copy because we want to original values
pd.concat((data_pred,y_pred_copy_df,Y_pred_prob), axis=1).to_csv(r"F:\Work\Upwork\Music Text Entity Matching\PredictedValues_NLP.csv")

