import joblib
import streamlit as st

# Load Model & Vectorizer
model = joblib.load("models/model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Tampilan Aplikasi
st.title("Aplikasi Analisis Sentimen Komentar - ファイサル")
st.write("Aplikasi ini digunakan untuk memprediksi apakah sebuah komentar bernada **positif** atau **negatif** menggunakan model Logistic Regression yang sudah dilatih sebelumnya.")

# Input dari user
komentar = st.text_input("Masukkan komentar yang ingin dianalisis")

# Tombol submit
if st.button("Submit"):
    if komentar.strip() == "":
        st.warning("Komentar tidak boleh kosong.")
    else:
        # Transform dan prediksi
        vector = tfidf.transform([komentar])
        prediksi = model.predict(vector)[0]

        label_map = {
            0: "Negatif",
            1: "Positif"
        }

        hasil = label_map.get(prediksi, "Tidak Dikenal")

        st.subheader("Hasil Analisis Sentimen")
        st.write(f"**Komentar:** {komentar}")
        st.write(f"**Prediksi Sentimen:** {hasil}")
