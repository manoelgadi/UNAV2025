
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, request

app = Flask(__name__)

from joblib import load
model = load('Rating_RandomForestClassifier.joblib')

@app.route('/')
def index():
    return render_template("index.html")


import pandas as pd

@app.route('/probdefaultfixed')
def probdefaultfixed():
    data = {
        'ebitda_income' : [0.3],
        'debt_ebitda' : [20],
        'rraa_rrpp' : [0.5],
        'log_operating_income' : [2000]
    }

    # creating a Dataframe object
    data_df = pd.DataFrame(data)
    X = data_df[['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income']]

    # predict
    prob_default = model.predict_proba(X)[:,1]

    return "Probability of default: {}".format(prob_default)



@app.route("/probdefault")
def probdefault():
    """ Probability of Default Harded Coded """
    # dictionary with list object in values
    data = {
        'ebitda_income' : [request.args["ebitda_income"]],
        'debt_ebitda' : [request.args["debt_ebitda"]],
        'rraa_rrpp' : [request.args["rraa_rrpp"]],
        'log_operating_income' : [request.args["log_operating_income"]]
    }

    # creating a Dataframe object
    data_df = pd.DataFrame(data)
    X = data_df[['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income']]

    # predict
    prob_default = model.predict_proba(X)[:,1]

    return "Probability of default: {}".format(prob_default)

