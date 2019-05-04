from sklearn.externals import joblib
import numpy as np

def model(q,ra,sa,mtype):
    # # add preprocessing calls here
    #
    # # load the saved model and return prediction
    # pred = ""
    inp_feat = np.random.rand(1,59)
    if mtype == "classifier":
        loaded_model = joblib.load("classifier.sav")
        pred = loaded_model.predict(inp_feat).ravel()[0]
        if pred >= 0.5:
            return "Correct"
        else:
            return "Incorrect"
    else:
        loaded_model = joblib.load("regressor.sav")
        pred = loaded_model.predict(inp_feat).ravel()
        return str(pred)
    return "Correct"
