# Text Classification
____________________________________________________________________________________________
This repo is a base repo for text classification. There are many steps involved in a full text classification project including -
1) Data Cleaning
2) Data splitting
3) Tf-Idf vectorisation
4) Model selection
5) Parameters selection
6) Model Saving
7) Prediction

This repo takes care of all the above steps in just 8-10 lines of code to produce a base model which in most cased would give you over 70% accuracy. With certain tweaks to paramaters and adding domain specific cleaning steps users can expect the accuracy to increase.


# Prerequisites for building a text classification model
There are two major requirements for a text classification model development - 

1) Text data that needs to be categorized

2) Labelled data - There is no thumb rule to identify as to how much labelled data is enough for a good model because it depends on number of classes to be predicted, variation in data for each class and likelihood of new patterns/texts to appear in future. Without any doubt, the more the labelled data we feed to model, the better would be the model accuracy.  However, some times it may be hard to have some labelled data at the beginning so there is no harm is trying to build a model if user have over 20 labelled documents for each category



# Process Steps Explanation
Data Cleaning - A text document may contain any possible words or letters in the world but a lot of words like prepositions (to, on, about, in, of etc.), helping verbs (am, are, were, was etc.), numbers and many other special characters are generally not useful for prediction of class of text and a model ideally should only focus on words that are helpful in defining the category of text. As a human, if we were to distinguish between a legal document and technology blog then most likely our focus would be on keywords rather than punctuation, verbs, helping verbs and prepositions etc. Similarly, we should reduce the word count be removing non-value adding words from the documents (email, book, letter or any text corpus)  to be classified so that model gets only relevant/value-adding parameters as an input. It helps model in two ways -

1) By reducing the number of parameters, the speed of model increases

2) We save the model from getting diluted by focusing only on keywords

Data Splitting into Training & Test sets  - This is done so we can test our model's performance. It is a good practice to train the model on 50%-70% of labelled data and test it on remaining depending on total size of labelled data. Again, there is no thumb rule but 70-30 splitting is generally a good practice when quantity of labelled data is not huge

Model Training - Model training is a major step in the process of model building wherein a type of classification model is chosen and trained using the train set created in the above step. There are some decisions that needs to be made to choose the right model, model parameters etc but i won't go into those details in this article. The in-house class created for text classification, takes care of all of it

Model Testing - Process of testing the above trained model on the test set is called model testing. Sometimes it can also be called as model validation, while there is a slight difference in both but i believe it is not important to highlight in this article. Model performance is measured using the results of model testing.

Please note that measuring only accuracy while building a classification model is not a good practice as it may be misleading in some unusual cases. A detailed confluence page Machine Learning - Test Strategy has been created by Tony Morrison which can be referred, should you wish to go in detail of various testing techniques and their interpretation

Cross Validation - Some times measuring performance on just one set of test data may give incorrect results because of non-symmetric distribution of train and test data so it is recommended to perform cross validation as well which essentially repeats the above two steps on different train-test combinations of labelled data. For more information on cross validation, please refer this link.

Model Saving - Once user is happy with model performance, they can choose to save the trained model and use it directly for prediction in future

Predictions - The saved model can be referred to make future classification predictions by using the predict ability of the model


# What can i expect in future versions?
___________________________________________________________________________________________
While this is very basic but effective model as of now, i look to add embeddings and deep learning options in future versions.

# Feedback mechanism
___________________________________________________________________________________________
Hope you find this useful. In case you face any challenges/errors, please do not forget to create new issues in github repo. 

