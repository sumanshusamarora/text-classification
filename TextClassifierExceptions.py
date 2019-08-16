# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:50:49 2019

@author: aroras13
"""

class TextClassificationExceptions(Exception):
    def __init__(self):
        super().__init__()
    pass
    
class IncorrectClassificationType(TextClassificationExceptions):
    pass


class IncorrectOverrideError(TextClassificationExceptions):
    pass

class EmailWrongFormatError(TextClassificationExceptions):
    pass

class IncorrectRandomSamplingError(TextClassificationExceptions):
    pass

class DataNotCorrect(TextClassificationExceptions):
    pass

         
        
