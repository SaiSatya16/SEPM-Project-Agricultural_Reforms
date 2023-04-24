from flask import Flask,render_template,redirect,request
from model import *
import numpy as np
import pandas as pd
import pickle
from ml_model_croprecom import *
from ml_model_fertilizerrecom import *
crop_recommendation_model_path = 'knn_pipeline.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

fertilizer_recommendation_model_path = 'rf_pipeline.pkl'
fertilizer_recommendation_model = pickle.load(
    open(fertilizer_recommendation_model_path, 'rb'))


#creating the app
app=Flask(__name__)

#configuring the app
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///agriculturedata.sqlite3"

@app.route('/',methods=['GET',"POST"])
def index_page():
    return render_template('index.html')

@app.route('/recommendation')
def rec_page():
    return render_template('recommendation.html')

@app.route('/fertilizer_rec')
def ferrec_page():
    return render_template('fertilizer_rec.html')

@app.route('/fert_rec',methods=['GET','POST'])
def fert_rec():
    if request.method == 'POST':
        f = [float(x) for x in request.form.values()]
        data1 = [np.array(f)]
        print(data1)
        my_prediction1 = fertilizer_recommendation_model.predict(data1)
        final_prediction = my_prediction1[0]
        final_prediction = fertname_dict[final_prediction]

        return render_template('fertilizer_rec.html', prediction_text=final_prediction)

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        f = [float(x) for x in request.form.values()]
        data = [np.array(f)]
        print(data)
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        final_prediction = label_dict[final_prediction]

        return render_template('recommendation.html', prediction_text=final_prediction)


@app.route('/foreign_crop',methods=['GET','POST'])
def foreign_crop():
    return render_template('foreign.html',data=[{'name':'Hop Shoots'}, {'name':'Cassava'}, {'name':'Buckwheat'},{'name':'Forage Sorghum'},{'name':'Brussel Sprouts'},{'name':'Camachile'},{'name':'Mangosteen'},{'name':'Avocado'},{'name':'Persimmon'},{'name':'Karonda'}])

@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    if select == 'Hop Shoots':
        return render_template('HopShoots.html')
    elif select == 'Cassava':
        return render_template('Cassava.html')
    elif select == 'Buckwheat':
        return render_template('Buckwheat.html')
    elif select == 'Forage Sorghum':
        return render_template('ForageSorghum.html')
    elif select == 'Brussel Sprouts':
        return render_template('BrusselSprouts.html')
    elif select == 'Camachile':
        return render_template('Camachile.html')
    elif select == 'Mangosteen':
        return render_template('Mangosteen.html')
    elif select == 'Avocado':
        return render_template('Avocado.html')
    elif select == 'Persimmon':
        return render_template('Persimmon.html')
    else:
        return render_template('Karonda.html')
    
    
    

@app.route('/price',methods=['GET','POST'])
def price_prediction():
    return render_template('price_prediction.html')

if __name__ == "__main__":
    app.run(debug=True)

