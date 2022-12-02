from flask import Flask, render_template,request
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

app = Flask(__name__)
model = pickle.load(open('deploy-nb-project/model2.pkl', 'rb'))

DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

X = data.iloc[:,:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test =train_test_split(
X, y, test_size = 0.2, random_state = 24)



@app.route("/")
# we have render_template so as to render HTML file everytime we go to the link
# for now which is "/" which means home page 

def hello():
    return render_template('index.html')




@app.route("/predict", methods=['POST'])

def predict():
    inpp = request.form['symptom']
    symptoms = X.columns.values

    symptom_index = {}
    for index,value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index
    
    data_dict = {
        "symptom_index":symptom_index,
    }

    inpp = inpp.split(",")
    inpd = [0] * len(data_dict["symptom_index"])

    for inp in inpp:
        index = data_dict["symptom_index"][inp]
        inpd[index] = 1
        inpd = np.array(inpd).reshape(1,-1)



    
    prediction = nb_model.predict(inpd)[0]
    return render_template('index.html', prediction_text=f'For the symptom {inpp} you might have ${prediction}')

    



# def predict():
#     inpp = request.form['symptom']
#     #inpp = "Itching"
#     symptoms = X.columns.values

#     symptom_index = {}
#     for index,value in enumerate(symptoms):
#         symptom = " ".join([i.capitalize() for i in value.split("_")])
#         symptom_index[symptom] = index
    
#     data_dict = {
#         "symptom_index":symptom_index,
#     }
    
    
#     def predictD(inp):
#         inp = inp.split(",")
#         inpd = [0] * len(data_dict["symptom_index"])

#         for inp in inp:
#             index = data_dict["symptom_index"][inp]
#             inpd[index] = 1
#         inpd = np.array(inpd).reshape(1,-1)

#         # print(model.predict(inpd)[0])
#         prediction = model.predict(inpd)[0]
#         return render_template('index.html', prediction_text=f'For the symptom {inpp} you might have ${prediction}')
    
#     predictD(inpp)
    
#----------------------------------------------------------------------------------------------------------------------------------

    #prediction = model.predict(symptom)



    #     return render_template('index.html', prediction_text=f'for the symptom {symptom} you might have ${prediction}')
    # predictD(inpp)







if __name__=="__main__":
    app.run()