# app.py
from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def predict():
    if request.method == 'POST':
        # Get form data from the request
        data = request.get_json()
        age = data['age']
        hypertension = data['hypertension']
        heart_disease = data['heart_disease']
        smoking_status = data['smoking_status']
        avg_glucose_level = data['avg_glucose_level']
        bmi = data['bmi']
        gender = data['gender']

        # Call the machine learning function to get the prediction
        prediction = predict(age, hypertension, heart_disease, smoking_status, avg_glucose_level, bmi, gender)

        # Return the prediction result as JSON
        return ({'prediction': prediction})
def predict():

    # IMPORTING THE LIBRARIES

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import warnings
    warnings.simplefilter('ignore')

    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    df = pd.read_csv("C:\\Users\\HPRIYA\\OneDrive\\Desktop\\healthcare-dataset-stroke-data.csv")


    df

    df.head()

    df.tail()

    df.shape

    df.describe()

    # DATA CLEANING

    df.info()

    df.isnull().sum()

    df['bmi'].fillna(df['bmi'].mean(),inplace = True)

    df

    df.isnull().sum()

    # DATA VISUALISATION

    import seaborn as sns
    sns.set()

    df['gender'].value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'gender' is a column in the DataFrame 'df'
    sns.countplot(x='gender', data=df)

    # Optional: If you want to display the plot


    df['ever_married'].value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'ever_married' is a column in the DataFrame 'df'
    sns.countplot(x='ever_married', data=df)

    # Optional: If you want to display the plot


    df['work_type'].value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'work_type' is a column in the DataFrame 'df'
    sns.countplot(x='work_type', data=df)

    # Optional: If you want to display the plot


    df['Residence_type'].value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'Residence_type' is a column in the DataFrame 'df'
    sns.countplot(x='Residence_type', data=df)

    # Optional: If you want to display the plot


    df['smoking_status'].value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'smoking_status' is a column in the DataFrame 'df'
    sns.countplot(x='smoking_status', data=df)

    # Optional: If you want to display the plot


    df['hypertension'].value_counts()
    #0 represents No Hypertension
    #1 represents Hypertension

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'hypertension' is a column in the DataFrame 'df'
    sns.countplot(x='hypertension', data=df)

    # Optional: If you want to display the plot


    df['heart_disease'].value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'heart_disease' is a column in the DataFrame 'df'
    sns.countplot(x='heart_disease', data=df)

    # Optional: If you want to display the plot


    df['stroke'].value_counts()
    #0 represents No Stroke
    #1 represents Stroke

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'stroke' is a column in the DataFrame 'df'
    sns.countplot(x='stroke', data=df)

    # Optional: If you want to display the plot


    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'gender' and 'stroke' are columns in the DataFrame 'df'
    sns.countplot(x='gender', hue='stroke', data=df)

    # Optional: If you want to display the plot


    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming 'gender' and 'hypertension' are columns in the DataFrame 'df'
    sns.countplot(x='gender', hue='hypertension', data=df)

    # Optional: If you want to display the plot



    df = df.drop(columns = ['ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis = 1)

    df

    df.replace({'gender' : {'Male' : 0 , 'Female' : 1 , 'Other' : 2}}, inplace = True)

    df

    #Seperating the data and labels
    X = df.drop(columns = ['gender', 'hypertension' , 'heart_disease', 'stroke'], axis = 1)
    Y_hypertension = df['hypertension']
    Y_heartdisease = df['heart_disease']
    Y_stroke = df['stroke']

    X

    #Data standardisation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    standard = scaler.transform(X)
    X = standard

    X

    Y_hypertension

    Y_heartdisease

    Y_stroke

    # SPLIT DATA IN TEST AND TRAIN FOR HYPERTENSION PREDICTION

    #Train,Test,Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_hypertension, test_size = 0.2, stratify = Y_hypertension, random_state = 2)

    from sklearn import svm
    model = svm.SVC(kernel = 'linear')

    #Training the SVM Model
    model.fit(X_train, Y_train)

    #Finding the accuracy score on train dataset
    from sklearn.metrics import accuracy_score
    X_train_prediction = model.predict(X_train)
    train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    train_data_accuracy

    #Finding the accuracy score on test dataset
    from sklearn.metrics import accuracy_score
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    test_data_accuracy

    # SPLIT DATA IN TEST AND TRAIN FOR STROKE PREDICTION

    #Train,Test,Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_stroke, test_size = 0.2, stratify = Y_stroke, random_state = 2)

    from sklearn import svm
    model = svm.SVC(kernel = 'linear')

    #Training the SVM Model
    model.fit(X_train, Y_train)

    #Finding the accuracy score on train dataset
    from sklearn.metrics import accuracy_score
    X_train_prediction = model.predict(X_train)
    train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    train_data_accuracy

    #Finding the accuracy score on test dataset
    from sklearn.metrics import accuracy_score
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    test_data_accuracy

    # MODEL EVALUATION FOR STROKE PREDICTION

    #Predicting System


    ht=int(input("Enter 1 if they have hypertension else 0: "))
    hd=int(input("Enter 1 if they have heart disease else 0: "))
    glu=float(input("Enter the average glucose level: "))
    bmi=float(input("Enter the BMI value: "))



    data = (ht,hd,glu,bmi)

    data_array = np.asarray(data)

    #Reshaping the array
    data_reshape = data_array.reshape(1, -1)

    #Standardizing the data
    data_standard = scaler.transform(data_reshape)

    prediction = model.predict(data_standard)

    if(prediction[0] == 0):
        return ('No Stroke')
    else:
        return ('Stroke')

pass
if __name__ == '__main__':
    app.run(debug=True)


