import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients.csv'
df = pd.read_csv(url)

# Prepare the data
X = df[['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']]  # Features
y = df['Output']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# For Streamlit app
st.title('Soil Fertility Analysis')
ph = st.number_input('Enter pH level:')
n = st.number_input('Enter Nitrogen level:')
p = st.number_input('Enter Phosphorus level:')
k = st.number_input('Enter Potassium level:')

ec = st.number_input('Enter Potassium level:')
oc = st.number_input('Enter Potassium level:')
s = st.number_input('Enter Potassium level:')
zn = st.number_input('Enter Potassium level:')
fe = st.number_input('Enter Potassium level:')
cu = st.number_input('Enter Potassium level:')
mn = st.number_input('Enter Potassium level:')
b = st.number_input('Enter Potassium level:')


# Add inputs for the rest of the features

if st.button('Analyze'):
    # Note: Add the rest of your feature inputs here in the correct order
    input_features = scaler.transform([[n, p, k, ph, pc,oc,s,zn,fe,cu,mn,b]])  # Fill in the rest of the inputs
    prediction = model.predict(input_features)
    st.write(f'The prediction is: {prediction[0]}')





















