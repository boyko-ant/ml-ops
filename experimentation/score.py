#!/usr/bin/env python
# coding: utf-8

# # Score Data with a Ridge Regression Model Trained on the Diabetes Dataset

# This notebook loads the model trained in the Diabetes Ridge Regression Training notebook, prepares the data, and scores the data.

# In[15]:


import json
import numpy
from azureml.core.model import Model
import joblib


# ## Load Model

# In[16]:


def init():
    model_path = Model.get_model_path(
        model_name="sklearn_regression_model.pkl")
    model = joblib.load(model_path)
    return model


# ## Prepare Data

# In[17]:


def run(raw_data, request_headers, model):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)

    return {"result": result.tolist()}


# ## Score Data

# In[18]:


model = init()
test_row = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
request_header = {}
prediction = run(test_row, request_header, model)
print("Test result: ", prediction)

