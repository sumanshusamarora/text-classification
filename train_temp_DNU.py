def train_model(self, X_train=None, y_train=None, Override_existing_X_y:bool = False, max_feat:int=900, inverse_regularisation:int = 70, Tf_Idf:bool=True, HyperTuning:bool=False, HyperTuning_params={}):
        if self.model_training == True:
            try:
                if self.X_train == None:
                    raise TextClassifierExceptions.IncorrectOverrideError
            except:
                raise TypeError("\033[0;31mTrain_test_split did not run correctly, please fix that first")
            logging.debug('----TextClassfier.py Model training block started----')
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
            Logistic_classifier = LogisticRegression(C=inverse_regularisation, dual=False, multi_class='ovr', penalty='l1', solver='saga', tol=0.001, max_iter=5000, random_state=self.RandomState)
            
            if HyperTuning == True:
                if len(HyperTuning_params) == 0:
                    raise TypeError("\033[0;31mHypertuning set to True but hyper parammeter grid not provided")
                else:
                    clf = LogisticRegression()
                    gcv = GridSearchCV(clf, param_grid=HyperTuning_params, scoring='accuracy', error_score=0.0, verbose=5)
                    if Tf_Idf == False:
                        vect_HT = CountVectorizer()
                        temp_X_HT = vect_HT.fit_transform(self.temp_X_train)
                    else:
                        Log_pipeline_HT = Pipeline([('vect_HT', TfidfVectorizer(ngram_range=(1,2), max_features=max_feat, stop_words='english')),
                                ('tfidf_HT', TfidfTransformer())
                                ])
                        temp_X_HT = Log_pipeline_HT.fit_transform(self.temp_X_train)
                    
                    GCV_Result = gcv.fit(temp_X_HT, self.temp_y_train)
                    Logistic_classifier = GCV_Result.best_estimator_
                    
            if Tf_Idf == False:
                Log_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('clf', Logistic_classifier)
                       ])
            else:
                Log_pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), max_features=max_feat, stop_words='english')),
                        ('tfidf', TfidfTransformer()),
                        ('clf', Logistic_classifier)
                       ])
            self.Log_pipeline = Log_pipeline.fit(self.temp_X_train, self.temp_y_train)
            try:
                if self.Log_pipeline == None:
                    self.temp_X_train = None
                    self.temp_y_train = None
                    raise TextClassifierExceptions.IncorrectOverrideError
                    
                
            except:
                raise TypeError("\033[0;31mPipeline not created successfully, please check your parameters")
            logging.debug('----TextClassfier.py Model training block finished----')
            return self.Log_pipeline
