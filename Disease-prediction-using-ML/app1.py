import streamlit as st
from auth import signup, login, logout, is_admin, set_admin
from PIL import Image
import base64
import pandas as pd
import sqlite3
from datetime import datetime
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Background image
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("data/disease-prediction.jpg")  

# Connect to DB
conn = sqlite3.connect('data/predictions_history.db', check_same_thread=False)
cursor = conn.cursor()

# Create history table with an added 'specialist' column
cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        symptoms TEXT,
        predicted_disease TEXT,
        confidence_score REAL,
        diet TEXT,
        workout TEXT,
        medication TEXT,
        precaution TEXT,
        specialist TEXT,
        model TEXT,
        timestamp TEXT
    )
''')
conn.commit()

# Load merged training data
training_data = pd.read_csv('data/Training.csv')
symptom_descriptions = pd.read_csv('data/symptoms_description.csv')  # Adjust path if needed
symptom_desc_map = dict(zip(symptom_descriptions['Symptom'], symptom_descriptions['Description']))

# Extract recommendation mapping (disease to details)
recommendation_map = training_data.drop_duplicates(subset='prognosis').set_index('prognosis')[['Description', 'Diet', 'Medication', 'Workout', 'Precaution', 'Specialist']]

# Prepare features
all_columns = training_data.columns.tolist()
non_symptom_cols = ['prognosis', 'Description', 'Diet', 'Medication', 'Workout', 'Precaution', 'Specialist']

symptoms = [col for col in training_data.columns if col not in non_symptom_cols]

X = training_data[symptoms]
y = training_data['prognosis']

# Load models
with open("decision_tree.pkl", "rb") as f:
    clf_dt = pickle.load(f)
with open("random_forest.pkl", "rb") as f:
    clf_rf = pickle.load(f)
with open("naive_bayes.pkl", "rb") as f:
    clf_nb = pickle.load(f)
with open("knn.pkl", "rb") as f:
    clf_knn = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Auth state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Auth UI
def authentication_ui():
    if not st.session_state.logged_in:
        option = st.sidebar.radio("Login / Signup", ["Login", "Signup"])
        if option == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                success, is_admin_flag = login(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.is_admin = is_admin_flag
                    st.rerun()
        else:
            username = st.sidebar.text_input("Choose Username")
            password = st.sidebar.text_input("Choose Password", type="password")
            if st.sidebar.button("Signup"):
                signup(username, password)
    else:
        st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
        if is_admin(st.session_state.username):
            st.session_state.is_admin = True
        if st.session_state.is_admin:
            new_admin = st.sidebar.text_input("Enter Username to Promote")
            if st.sidebar.button("Promote"):
                if set_admin(new_admin):
                    st.success(f"{new_admin} is now an admin!")
                    st.rerun()
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()

authentication_ui()

# UI
st.title("🦠 Disease Prediction System")

if st.session_state.logged_in:
    st.subheader("🩺 Select Symptoms:")

    symptom_inputs = []
    for i in range(5):
        selected_symptom = st.selectbox(f"Symptom {i+1}", [''] + symptoms, key=f"symptom_{i}")
        symptom_inputs.append(selected_symptom)

        # Show description immediately after selection
        if selected_symptom:
            desc = symptom_desc_map.get(selected_symptom, "Description not available")
            st.markdown(f"<div style='margin-top: -10px; color: #444;'>📝 <strong>{selected_symptom}</strong>: {desc}</div>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model", ["", "Decision Tree", "Random Forest", "Naive Bayes", "KNN"])

    if st.button("🚀 Predict"):
        selected_symptoms = [s for s in symptom_inputs if s]
        if selected_symptoms and selected_model:
            input_vector = [1 if s in selected_symptoms else 0 for s in symptoms]
            input_df = pd.DataFrame([input_vector], columns=symptoms)

            # Model Selection
            if selected_model == "Decision Tree":
                model = clf_dt
            elif selected_model == "Random Forest":
                model = clf_rf
            elif selected_model == "Naive Bayes":
                model = clf_nb
            elif selected_model == "KNN":
                model = clf_knn

            # Model Prediction and Confidence
            # Model Prediction and Confidence
            pred = model.predict(input_df)[0]
            predicted_disease = label_encoder.inverse_transform([pred])[0]
            confidence = model.predict_proba(input_df)[0].max() * 100



            # Get Recommendations
            if predicted_disease in recommendation_map.index:
                row = recommendation_map.loc[predicted_disease]
                description = row['Description']
                diet = row['Diet']
                medication = row['Medication']
                workout = row['Workout']
                precaution = row['Precaution']
                specialist = row['Specialist']

                st.markdown(f"### 📝 Disease Description:\n**{description}**")

                rec_df = pd.DataFrame({
                    "Category": ["Diet", "Workout", "Medication", "Precaution"],
                    "Recommendation": [diet, workout, medication, precaution]
                })
                st.markdown("### 🍎 Recommendations:")
                st.dataframe(rec_df, use_container_width=True)

                st.markdown(f"### 🩻 Recommended Specialist:\n**{specialist}**")
                
                # Display Confidence Score
                st.markdown(f"### 🧑‍⚕️ Confidence Score: **{confidence:.2f}%**")

                # Save to DB with model used and confidence score
                # Save to DB with model used and confidence score
                cursor.execute('''
                    INSERT INTO history (username, symptoms, predicted_disease, confidence_score, diet, workout, medication, precaution, specialist, model, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (st.session_state.username, ', '.join(selected_symptoms), predicted_disease, confidence, diet, workout, medication, precaution, specialist, selected_model, datetime.now()))
                conn.commit()


                st.success(f"Prediction and recommendations for {predicted_disease} have been saved successfully!")

            else:
                st.error(f"No recommendations found for {predicted_disease}")

# History
if st.session_state.logged_in and st.button("📜 View Prediction History"):
    if is_admin(st.session_state.username):
        history_df = pd.read_sql_query("SELECT * FROM history", conn)
        st.markdown("### 🔍 All User History (Admin)")
    else:
        history_df = pd.read_sql_query("SELECT * FROM history WHERE username = ?", conn, params=(st.session_state.username,))
        st.markdown("### 📜 Your Prediction History")

    if not history_df.empty:
        st.dataframe(history_df)

        # Charts
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

        trend_data = history_df.groupby('predicted_disease').size().reset_index(name='count')
        fig_trend = px.bar(trend_data, x="predicted_disease", y="count", color="count",
                           title="Most Common Predicted Diseases", color_continuous_scale="Viridis")
        st.plotly_chart(fig_trend)

        symptom_counts = pd.Series(', '.join(history_df['symptoms']).split(', ')).value_counts().reset_index()
        symptom_counts.columns = ['Symptom', 'Count']
        fig_symptom = px.bar(symptom_counts, x="Symptom", y="Count", color="Count", 
                             title="Symptom Frequency in History", color_continuous_scale="YlOrRd")
        st.plotly_chart(fig_symptom)
        
        # ✅ Predicted Disease Distribution (Pie Chart)
        if not history_df.empty:
            disease_counts = history_df['predicted_disease'].value_counts()
            fig_disease_dist = px.pie(disease_counts, names=disease_counts.index, values=disease_counts.values,
                                    title="Predicted Disease Distribution")
            st.plotly_chart(fig_disease_dist)
    else:
        st.warning("No history available yet.")

# ✅ Close connection
conn.close()
