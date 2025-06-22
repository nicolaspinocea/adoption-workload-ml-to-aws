# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback
import flask
import pandas as pd
import re

print('libraries loaded')

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

class ScoringService(object):
    
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path,'model-ligthgbm.pkl'), 'rb') as inp: #'decision-tree-model.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model
    print('model loaded')
    @classmethod
    def predict(cls,input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        print('get model')

        model = cls.get_model()
        x=input
        return model.predict(x)   
 

# The flask app for serving predictions

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])


def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    print('start transformation')
    data = None
    print('start read data inference')
    print(flask.request.content_type)
    # Convert from CSV to pandas
    if flask.request.content_type and flask.request.content_type.startswith('text/csv'):
        #flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        print(data)
        s = io.StringIO(data)
        print(s)
        data = pd.read_csv(s, header=None)#,encoding='utf-8',sep=';',)
        print(data)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    print('do the prediction')

    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')