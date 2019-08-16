import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import time
import numpy as np
from rake_nltk import Rake
r = Rake()
import heapq
import pandas as pd

#Defining custom stopwords list
customer_stopwords = ['account', 'a/c', 'acc', 'acct', 'act', 'bsb', 'name', 'thanks', 'thank you', 'weekday',
                      'thanking', 'dear', 'mate', 'legally', 'privileged', 'consider', 'computer' 'virus', 'data', 'corruption',
                      'communication', 'error', 'notify', 'sender', 'view', 'views', 'expressed', 'express', 'hope', 'well',
                      'mailto', 're', 'www', 'com', 'coau', 'co', 'website',  'team', 'hi', 'hello', 'morning', 'noon', 'day',
                      'afternoon', 'evening', 'attached', 'night', 'long', 'anz', 'subject', 'dear', 'please', 'further', 'http', 
                      'https', 'twitter', 'facebook', 'youtube', 'instagram', 'addressee', 'copyright', 'confidentital', 
                      'confidentiality', 'important', 'notice', 'great', 'australia', 'new zealand', 'find', 'thank', 'fine',
                      'attachments', 'email', 'confirm', 'yet', 'detail', 'unless', 'intended', 'intent', 'wondering', 'would', 
                      'back', 'appear', 'thankyou', 'therefore', 'staff', 'question', 'answer', 'reason', 'south', 'north', 
                      'east', 'west', 'road', 'will', 'shall', 'should', 'would', 'may', 'might', 'lane', 'hwy', 'weekend',
                      'highway', 'address', 'highway', 'unit', 'house', 'home', 'again', 'looking', 'forward', 'classification',
                      'unable', 'able', 'store', 'shop', 'street', 'suburb', 'time', 'nsw', 'new south wales', 'ltd',
                      'qld', 'queensland', 'vic', 'victoria', 'canada', 'usa', 'china', 'singapore', 'india', 'brisbane', 'perth', 'south',
                      'square', 'avenue', 'ave', 'city', 'notification', 'kindly', 'regards', 'regard', 'best', 'none', 'collins',
                      'discuss', 'discussed', 'urgent', 'hurry', 'fast', 'otherwise', 'zealand', 'shanghai', 'melbourne', 'print', 
                      'printing', 'sydney', 'kent', 'level', 'floor' ,'environment', 'e-mail', 'message', 'receipient', 
                      'recipient', 'take', 'note', 'anti', 'aust', 'trust', 'job', 'number']

#Dunction to clean the corpus
def create_clean_corpus(X_Series):
    X_Series = pd.Series(X_Series)
    corpus = []
    counter = 1
    counter_overall = 1
    time_counter = time.time()   
    WNL = WordNetLemmatizer()
    PST = PorterStemmer()
    for i in range(0, len(X_Series)):
        if counter_overall == 3:
            time3 = time.time()    
        if counter_overall == 4:
            time4 = time.time()
            print("Corpus cleaning started... This might take approx " +  str("{0:.0f}".format((time4-time3)*len(X_Series)//60)) + " minute(s) and " +  str("{0:.0f}".format((time4-time3)*len(X_Series)%60)) + " second(s) to finish")  
        if i > 0 and i/1000 == i//1000:
            print("Batch " + str(counter) + " of 1000 rows out of " + str(len(X_Series)) + " rows finished cleaning in " + str("{0:.2f}".format(time.time() - time_counter)) + " seconds")
            counter+=1    
            time_counter = time.time()
        if i == len(X_Series)//2:
            print("Corpus cleaning is half way through now..")    
        #print(i)
        text_desc = X_Series[i]
        text_desc = text_desc.lower()
        text_desc = re.sub('[*/\%^!)()]', ' ', text_desc)
        text_desc = re.sub(r'^http?:\/\/.*[\r\n]*', ' ', text_desc, flags=re.MULTILINE)
        try:
            lst_TBR = re.findall('\S+@\S+', text_desc)
            if len(lst_TBR) > 0:
                for email_id in lst_TBR:
                    text_desc.replace(email_id, ' ')
        except:
            pass
        
        #text_desc = re.sub(r'\b[0-9]+\b',"*Number*", text_desc)
        text_desc = re.sub('[^a-zA-Z0-9]', ' ', text_desc)
        text_desc = text_desc.replace('\t', ' ')
        text_desc = text_desc.lower()        
        TB = TextBlob(text_desc)
        for x in range(len(TB.tags)):
            if TB.tags[x][1] == 'NNP' or TB.tags[x][1] == 'NNPS' or TB.tags[x][1] == 'UH':
                text_desc.replace(TB.tags[x][0], '')
                
        text_desc = text_desc.split()
        text_desc = [word for word in text_desc if not word in set(stopwords.words('english'))]
        text_desc = [word for word in text_desc if not word in set(r.stopwords + customer_stopwords)]
        text_desc = [word for word in text_desc if not word in r.to_ignore]
        #text_desc = [word for word in text_desc if not word in customer_stopwords]
        text_desc = [word for word in text_desc if len(word) > 2]
        text_desc = [WNL.lemmatize(word,'v') for word in text_desc]
        text_desc = [PST.stem(word) for word in text_desc]
        #text_desc = [word.replace('bank', '') for word in text_desc]
        text_desc = ' '.join(text_desc)   
        corpus.append(text_desc)
        counter_overall+=1 
    print("Corpus cleaning finished!")
    return corpus

#Dunction to clean the corpus
def clean_small_corpus(X_Series):
    X_Series = pd.Series(X_Series)
    corpus = []
    counter = 1
    counter_overall = 1
    time_counter = time.time()   
    #WNL = WordNetLemmatizer()
    #PST = PorterStemmer()
    for i in range(0, len(X_Series)):
        if counter_overall == 3:
            time3 = time.time()    
        if counter_overall == 4:
            time4 = time.time()
            print("Corpus cleaning started... This might take approx " +  str("{0:.0f}".format((time4-time3)*len(X_Series)//60)) + " minute(s) and " +  str("{0:.0f}".format((time4-time3)*len(X_Series)%60)) + " second(s) to finish")  
        if i > 0 and i/1000 == i//1000:
            print("Batch " + str(counter) + " of 1000 rows out of " + str(len(X_Series)) + " rows finished cleaning in " + str("{0:.2f}".format(time.time() - time_counter)) + " seconds")
            counter+=1    
            time_counter = time.time()
        if i == len(X_Series)//2:
            print("Corpus cleaning is half way through now..")    
        #print(i)
        text_desc = X_Series[i]
        text_desc = text_desc.lower()
        text_desc = re.sub('[*/\%^!)()]', ' ', text_desc)
        text_desc = re.sub(r'^http?:\/\/.*[\r\n]*', ' ', text_desc, flags=re.MULTILINE)
        try:
            lst_TBR = re.findall('\S+@\S+', text_desc)
            if len(lst_TBR) > 0:
                for email_id in lst_TBR:
                    text_desc.replace(email_id, ' ')
        except:
            pass
        
        #text_desc = re.sub(r'\b[0-9]+\b',"*Number*", text_desc)
        text_desc = text_desc.replace('\t', ' ')
        text_desc = text_desc.lower()            
        text_desc = re.sub('[^a-z0-9]', ' ', text_desc)            
        text_desc = text_desc.split()
        #text_desc = [WNL.lemmatize(word,'v') for word in text_desc]
        #text_desc = [PST.stem(word) for word in text_desc]
        #text_desc = [word.replace('bank', '') for word in text_desc]
        text_desc = ' '.join(text_desc)   
        corpus.append(text_desc)
        counter_overall+=1 
    print("Corpus cleaning finished!")
    return corpus


def remove_disclaimer_from_body(message):
    split_line_message = [line.lower() for line in [line for line in message.replace('\r', '').split("\n")] if (line.strip() not in [" ",  ""])]
    #Ading a special check to not make any changes if text startes with [ as this is old text and the format is not appropriate
    if split_line_message[0][0] != '[':
    #Checking if Conf_intend exist in same line
        index_lst_recpient_intend = [i for i, line in enumerate(split_line_message) if (("email" in line or "message" in line or 'e-mail' in line ) and ("receipient" in line or 'recipient' in line)  and "intended" in line)]
        if len(index_lst_recpient_intend) > 0:
            Recp_Intend_Found_len = len(index_lst_recpient_intend)
            Recp_Intend_email_list = list(np.array(split_line_message)[index_lst_recpient_intend])
        else:
            index_lst_recpient_intend.append(100000)
            Recp_Intend_Found_len = 0
            Recp_Intend_email_list = []
    
        #Checking if Conf_intend_recepient in same line
        index_lst_conf_intend = [i for i, line in enumerate(split_line_message) if (("email" in line or "message" in line or 'e-mail' in line ) and "confidential" in line)]
        if len(index_lst_conf_intend) > 0:
            Conf_Intend_Found_len = len(index_lst_conf_intend)
            Conf_Intend_email_list = list(np.array(split_line_message)[index_lst_conf_intend])
        else:
            index_lst_conf_intend.append(100000)
            Conf_Intend_Found_len = 0
            Conf_Intend_email_list = []
    
        #Checking if Print_Envmnt exist in same line
        index_lst_Print_Envmt = [i for i, line in enumerate(split_line_message) if (("email" in line or "message" in line or 'e-mail' in line ) and "environment" in line and "print" in line)]
        if len(index_lst_Print_Envmt) > 0:
            Print_Envmt_Found_len = len(index_lst_Print_Envmt)
            Print_Envmt_email_list = list(np.array(split_line_message)[index_lst_Print_Envmt])
        else:
            index_lst_Print_Envmt.append(100000)
            Print_Envmt_Found_len = 0
            Print_Envmt_email_list = []
    
        #Checking if FROM: exist in email
        index_lst_FROM = [i for i, line in enumerate(split_line_message) if line[:5] == "from:"]
        if len(index_lst_FROM) > 0:
            From_Found_len = len(index_lst_FROM)
            From_email_list = list(np.array(split_line_message)[index_lst_FROM])
        else:
            index_lst_FROM.append(100000)
            From_Found_len = 0
            From_email_list = []
            
        #Scenario when there is no reply to the email
        try:
            if (Recp_Intend_Found_len > 0 or Conf_Intend_Found_len > 0 or Print_Envmt_Found_len > 0) and From_Found_len == 0:
                del split_line_message[min(min(index_lst_Print_Envmt), min(index_lst_conf_intend), min(index_lst_recpient_intend)):]
    
            #Scenario where a reply was received for email, deleting only last instance
            if (Recp_Intend_Found_len > 0 or Conf_Intend_Found_len > 0 or Print_Envmt_Found_len > 0) and From_Found_len > 0:
                match_list = [x for x in heapq.nlargest(2, [max(index_lst_Print_Envmt), max(index_lst_conf_intend), max(index_lst_recpient_intend)]) if x != 100000]
                if max(match_list) > max(index_lst_FROM):
                    del split_line_message[max(match_list):]
    
            #Scenario where a reply was received for email, deleting only second last instance
                if max(index_lst_FROM) - max(match_list) <=8:
                    del split_line_message[max(match_list):max(index_lst_FROM)]
        except:
            pass
        
        #Other known disclaimer sequences
        line_legally_privileged = [line for i, line in enumerate(split_line_message) if "legally privileged" in line]
        line_virus = [line for i, line in enumerate(split_line_message) if "computer virus, data corruption" in line]
        line_comm_error = [line for i, line in enumerate(split_line_message) if "communication in error" in line]
        line_Notify_sender = [line for i, line in enumerate(split_line_message) if "notify the sender" in line]
        line_printing = [line for i, line in enumerate(split_line_message) if "before printing" in line]
        index_views_expressed = [line for i, line in enumerate(split_line_message) if "views expressed" in line]
        thank_coop_expressed = [line for i, line in enumerate(split_line_message) if ("thank you" in line or "thanks" in line) and ('cooperation' in line or 'co-operation' in line)] 
        
        #Combining all removal sequences
        removal_lines_list = [Recp_Intend_email_list, Conf_Intend_email_list, Print_Envmt_email_list, line_legally_privileged,
                             line_virus, line_comm_error, line_Notify_sender, line_printing, index_views_expressed, thank_coop_expressed]
        
        #Removing sequences
        for list_type in removal_lines_list:
            split_line_message = [line for line in split_line_message if line not in list_type]

          
    return "\n".join(split_line_message)