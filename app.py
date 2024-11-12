from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash

from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
#loading models
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management and flash messages

# Dummy user database for demonstration (use a real DB in production)
users_db = {
    "sgtbmit@gmail.com": {"password": "123"}  # Example username/password
}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']  # Get the username from form
        password = request.form['password']  # Get the password from form
        remember_me = 'remember_me' in request.form  # Check if "Remember Me" is checked

        # Check if user exists in the dummy database
        if username in users_db and users_db[username]['password'] == password:
            session['username'] = username  # Store username in session
            flash('Login successful!', 'success')  # Flash a success message
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials, please try again.', 'danger')

    return render_template('index1.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')
    return f"Welcome to the dashboard, {session['username']}!"


@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Item_Indentifier = request.form["Item_Identifier"]
        Item_weight = request.form["Item_weight"]
        Item_Fat_Content = request.form["Item_Fat_Content"]
        Item_visibility = request.form["Item_visibility"]
        Item_Type = request.form["Item_Type"]
        Item_MPR = request.form["Item_MPR"]
        Outlet_identifier = request.form["Outlet_identifier"]
        Outlet_established_year = request.form["Outlet_established_year"]
        Outlet_size = request.form["Outlet_size"]
        Outlet_location_type = request.form["Outlet_location_type"]
        Outlet_type = request.form["Outlet_type"]


        features = np.array([[Item_Indentifier,Item_weight,Item_Fat_Content,Item_visibility,Item_Type,Item_MPR,Outlet_identifier,Outlet_established_year, Outlet_size, Outlet_location_type,Outlet_type]],dtype=np.float32)
        transformed_feature = features.reshape(1, -1)
        prediction = model.predict(transformed_feature)[0]
        print(prediction)
        return render_template('index.html',prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
