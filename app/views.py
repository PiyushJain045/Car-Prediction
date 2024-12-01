# views.py
import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render

# Load the trained model
model = pickle.load(open('app/LinearRegressionModel.pkl', 'rb'))

def home(request):
    if request.method == "POST":
        # Extract user input from the form
        name = request.POST['name']
        company = request.POST['company']
        year = int(request.POST['year'])
        kms_driven = int(request.POST['kms_driven'])
        fuel_type = request.POST['fuel_type']

        # Create a DataFrame for prediction
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                  data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5))

        # Make prediction
        prediction = model.predict(input_data)[0]

        prediction = round(prediction, 2)

        # Render result
        return render(request, 'home.html', {'prediction': prediction})

    return render(request, 'home.html')
