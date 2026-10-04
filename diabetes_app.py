import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("Diabetes Prediction App")
st.write(
    "Bu uygulama, hastanın sağlık ve demografik metriklerini kullanarak Gradient Boosting modeli ile diyabet riskini tahmin eder."
)


@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")


model = load_model()

st.subheader("Hasta Bilgilerini Giriniz:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yas (Age)", min_value=1.0, max_value=100.0, value=35.0)
    hypertension = st.selectbox("Hipertansiyon (Hypertension)", [0, 1])
    heart_disease = st.selectbox("Kalp Hastaligi (Heart Disease)", [0, 1])
    bmi = st.number_input("BMI (Vucut Kitle Indeksi)", min_value=10.0, max_value=70.0, value=25.0)

with col2:
    hba1c = st.number_input("HbA1c Seviyesi", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose = st.number_input("Kan Glikoz Seviyesi", min_value=50.0, max_value=300.0, value=100.0)
    gender = st.selectbox("Cinsiyet (Gender)", ["Female", "Male"])
    smoking_history = st.selectbox("Sigara Gecmisi (Smoking History)", ["never", "current", "former", "not current"])

if st.button("Diyabet Tahmini Yap", type="primary"):
    try:
        # Modelin egitildigi kolon yapisina uygun girdi cercevesi olusturma
        input_data = pd.DataFrame({
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [blood_glucose],
            "gender_Male": [1 if gender == "Male" else 0],
            "smoking_history_current": [1 if smoking_history == "current" else 0],
            "smoking_history_former": [1 if smoking_history == "former" else 0],
            "smoking_history_not current": [1 if smoking_history == "not current" else 0],
        })

        # Eksik kolonlari sifirliyoruz
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)
        result = "Yuksek Diyabet Riski (Pozitif)" if prediction[0] == 1 else "Diyabet Riski Dusuk (Negatif)"
        st.success(f"Tahmin Sonucu: **{result}**")
    except Exception as e:
        st.error(f"Tahmin sirasinda bir hata olustu: {e}")