#Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset
url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients.csv'
df = pd.read_csv(url)

# Preparing the data with the correct nutrient columns
X = df[['N', 'P', 'K', 'pH', 'Zn', 'Fe', 'EC', 'OC', 'S', 'Cu', 'Mn', 'B']]
y = df['Output']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA to identify 6 major nutrients
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# Calculating total importance of each feature across all PCA components
total_importance = np.abs(pca.components_).sum(axis=0)
sorted_indices = np.argsort(total_importance)[::-1]
top_features = X.columns[sorted_indices][:6].tolist()

# Training a RandomForestClassifier on the PCA-reduced dataset
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

#Streamlit Title
st.title('Soil Fertility Analysis')

# Creating input fields for the top 6 features only. dynamically based on the PCA components
input_values = {}
for feature in top_features:
    input_values[feature] = st.number_input(f'Enter {feature} level:', key=feature)

# Button to perform analysis
if st.button('Analyze'):
    # Prepare user input for prediction
    user_input_array = np.zeros(X.shape[1])
    for i, feature in enumerate(X.columns):
        if feature in input_values:
            user_input_array[i] = input_values[feature]
    user_input = user_input_array.reshape(1, -1)
    # Scaling the user input
    input_scaled = scaler.transform(user_input)
    #Transforming user input with PCA
    input_pca = pca.transform(input_scaled)

    # Making the prediction
    prediction = model.predict(input_pca)
    result = "High Fertile" if prediction[0] == 1 else "Low Fertile"
    st.write(f'The prediction is: {result}')

    # Providing feedback based on the thresholds
    for feature in top_features:
        feature_value = input_values[feature]
        feedback_message = f'{feature} level is {feature_value}. '

        #Nitrogen
        if feature.startswith('N'):
            feedback_message = f'Nitrogen level is {feature_value}. '
            if feature_value < 280:
                feedback_message += "Nitrogen is deficient, which may leads to stunted growth and yellowing of leaves."
            elif feature_value > 560:
                feedback_message += "Nitrogen level is high,can result in excessive foliage growth at the expense of fruit and flower development, and can also cause nitrogen burn."
            else:
                feedback_message += "The Nitrogen level is optimal for the crop, promoting improved yield."

        #Phosphorous
        if feature.startswith('P'):
            feedback_message = f'Phosphorous level is {feature_value}. '
            if feature_value < 22.5:
                feedback_message += "Phosphorus is deficient, results in stunted growth and dark, dull, or purple-tinged leaves."
            elif feature_value > 55:
                feedback_message += "Phosphorus level is high, rare, but can inhibit the uptake of other nutrients like zinc and iron."
            else:
                feedback_message += "The Phosphorus level is optimal for the crop, promoting improved yield."

        #Potassium
        if feature.startswith('K'):
            feedback_message = f'Potassium level is {feature_value}. '
            if feature_value < 140:
                feedback_message += "Potassium is deficient, causes yellowing at leaf edges, weak stems, and reduced growth."
            elif feature_value > 330:
                feedback_message += "Potassium level is high, can lead to magnesium deficiency and salt stress in plants."
            else:
                feedback_message += "The Potassium level is optimal for the crop, promoting improved yield."

        #pH
        if feature.startswith('p'):
            feedback_message = f'ph level is {feature_value}. '
            if feature_value < 5.5:
                feedback_message += "ph is deficient, Nutrient solubility can increase to toxic levels; aluminum and manganese may become toxic."
            elif feature_value > 7.5:
                feedback_message += "ph level is high, Reduced solubility of micronutrients like iron, leading to deficiencies."
            else:
                feedback_message += "The ph level is optimal for the crop, promoting improved yield."

        #Zinc
        if feature.startswith('Z'):
            feedback_message = f'Zinc level is {feature_value}. '
            if feature_value < 0.6:
                feedback_message += "Zinc is deficient, Leads to stunted growth, smaller leaves, and interveinal chlorosis."
            elif feature_value > 1.5:
                feedback_message += "Zinc level is high, Can inhibit plant growth and reduce iron absorption"
            else:
                feedback_message += "The Zinc level is optimal for the crop, promoting improved yield."

        #Iron
        if feature.startswith('Fe'):
            feedback_message = f'Fe level is {feature_value}. '
            if feature_value < 0.6:
                feedback_message += "Iron is deficient, Causes chlorosis in young leaves, reduced flowering and fruiting."
            elif feature_value > 1.5:
                feedback_message += "Iron level is high,Rare in plants, but can cause bronzing and chlorosis."
            else:
                feedback_message += "The Iron level is optimal for the crop, promoting improved yield."

        #EC Electrical Conductivity
        if feature.startswith('E'):
            feedback_message = f'EC level is {feature_value}. '
            if feature_value < 200:
                feedback_message += "EC level is deficient, Indicates insufficient nutrients in the soil."
            elif feature_value > 1600:
                feedback_message += "EC level is high,Can lead to nutrient toxicity and osmotic stress, affecting plant water uptake."
            else:
                feedback_message += "The EC level is optimal for the crop, promoting improved yield."

        #Organic Carbon
        if feature.startswith('O'):
            feedback_message = f'OC level is {feature_value}. '
            if feature_value < 0.4:
                feedback_message += "OC level is deficient,results in poor soil structure and reduced water retention."
            elif feature_value > 3:
                feedback_message += "OC level is high,Generally beneficial but can lead to imbalances if not matched with nutrient availability."
            else:
                feedback_message += "The OC level is optimal for the crop, promoting improved yield."

        #Sulphur
        if feature.startswith('S'):
            feedback_message = f'Sulphur level is {feature_value}. '
            if feature_value < 10:
                feedback_message += "Sulphur level is deficient,Causes yellowing of leaves, similar to nitrogen deficiency."
            elif feature_value > 20:
                feedback_message += "Sulphur level is high,Rare, but can lead to decreased growth and enzyme inhibition."
            else:
                feedback_message += "The Sulphur level is optimal for the crop, promoting improved yield."

        #Copper
        if feature.startswith('C'):
            feedback_message = f'Copper level is {feature_value}. '
            if feature_value < 0.2:
                feedback_message += "Copper level is deficient,Leads to wilting, repressed growth, and chlorosis."
            elif feature_value > 5:
                feedback_message += "Copper level is high, can cause toxicity symptoms like leaf chlorosis and stunted growth."
            else:
                feedback_message += "The Copper level is optimal for the crop, promoting improved yield."

        #Manganese
        if feature.startswith('M'):
            feedback_message = f'Manganese level is {feature_value}. '
            if feature_value < 2:
                feedback_message += "Manganese level is deficient, results in interveinal chlorosis, necrotic spots, and reduced growth."
            elif feature_value > 4:
                feedback_message += "Manganese level is high, can lead to iron deficiency symptoms and reduced growth."
            else:
                feedback_message += "The Manganese level is optimal for the crop, promoting improved yield."
        #Boron
        if feature.startswith('B'):
            feedback_message = f'Boron level is {feature_value}. '
            if feature_value < 2:
                feedback_message += "Boron level is deficient, causes death of meristem tissue, brittle leaves, and stunted growth."
            elif feature_value > 4:
                feedback_message += "Boron level is high, Leads to leaf burn, chlorosis, and necrosis."
            else:
                feedback_message += "The Boron level is optimal for the crop, promoting improved yield."

        st.write(feedback_message)
