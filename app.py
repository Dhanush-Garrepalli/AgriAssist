import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients.csv'
df = pd.read_csv(url)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

st.title('Soil Fertility Analysis')
ph = st.number_input('Enter pH level:')
n = st.number_input('Enter Nitrogen level:')
p = st.number_input('Enter Phosphorus level:')
k = st.number_input('Enter Potassium level:')

if st.button('Analyze'):
    input_features = scaler.transform([[ph, n, p, k]])
    prediction = model.predict(input_features)
    st.write(f'The prediction is: {prediction[0]}')
