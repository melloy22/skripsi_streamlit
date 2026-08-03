"""
Aplikasi Prediksi Status Kelulusan Semester Mahasiswa
Program Studi Informatika - Universitas Internasional Semen Indonesia

Model  : SVM Tuned (Random Search) (tuned_svm_random_search_model.pkl)
Fitur  : KUMULATIF SKS, KUMULATIF IPK, KUMULATIF POINT SKEM
Target : LULUS / TIDAK LULUS

Catatan:
    Form input sekarang mencakup SEMUA kolom yang ada di file Excel
    "Riwayat Studi per Semester" (SKS, IPK, POINT SKEM per semester,
    Bobot, serta ketiga kolom kumulatif), supaya tampilannya sesuai
    dengan struktur data Excel Anda. Namun yang benar-benar dikirim
    ke model untuk prediksi tetap hanya 3 kolom kumulatif, sesuai
    fitur yang dipakai saat training (FEATURE_COLUMNS di bawah).

Cara menjalankan:
    streamlit run app.py

Pastikan file 'Tuned_SVM_RandomSearch_With_TomekLinks.pkl' berada di folder yang sama dengan app.py
"""

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "Tuned_SVM_RandomSearch_With_TomekLinks.pkl"

# PENTING: model ini (SVC) dilatih langsung dengan label berupa TEKS
# ('LULUS' / 'TIDAK LULUS'), BUKAN diencode jadi 0/1 lewat LabelEncoder.
# Jadi model.predict() sudah mengembalikan teks label secara langsung,
# tidak perlu dan tidak boleh dipetakan lewat LABEL_MAP versi angka.

# Urutan kolom fitur HARUS SAMA PERSIS dengan urutan saat training
FEATURE_COLUMNS = ["KUMULATIF SKS", "KUMULATIF IPK", "KUMULATIF POINT SKEM"]


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ============================================================
# HEADER
# ============================================================
st.title("🎓 Prediksi Status Kelulusan Semester Mahasiswa")
st.markdown(
    "Program Studi Informatika — Universitas Internasional Semen Indonesia  \n"
    "Model: **Tuned_SVM_RandomSearch_With_TomekLinks.pkl**"
)
st.divider()

if not model_loaded:
    st.error(
        f"File model `{MODEL_PATH}` tidak ditemukan. "
        "Pastikan file tersebut berada satu folder dengan `app.py`."
    )
    st.stop()

# ============================================================
# FORM INPUT (mengikuti struktur kolom Excel "Riwayat Studi per Semester")
# ============================================================
st.subheader("Input Data Mahasiswa")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
    #     nama_mahasiswa = st.text_input(
    #         "Nama / NIM Mahasiswa", placeholder="Contoh: Budi Santoso / 2021xxxxx"
    #     )
         semester = st.number_input("Semester", min_value=1, max_value=14, value=1, step=1)

    with col2:
        kumulatif_sks = st.number_input(
            "KUMULATIF SKS", min_value=0, max_value=200, value=0, step=1
        )
        kumulatif_ipk = st.number_input(
            "KUMULATIF IPK", min_value=0.0, max_value=4.0, value=0.0, step=0.01, format="%.2f"
        )
        kumulatif_skem = st.number_input(
            "KUMULATIF POINT SKEM", min_value=0, max_value=2000, value=0, step=10
        )

    submitted = st.form_submit_button("🔍 Prediksi Status Kelulusan", use_container_width=True)

# ============================================================
# PROSES PREDIKSI
# ============================================================
if submitted:
    # if not nama_mahasiswa.strip():
    #     st.warning("Mohon isi Nama / NIM Mahasiswa terlebih dahulu.")
    #     st.stop()

    # Fitur yang dikirim ke model HANYA 3 kolom kumulatif (sesuai training)
    input_df = pd.DataFrame(
        [[kumulatif_sks, kumulatif_ipk, kumulatif_skem]],
        columns=FEATURE_COLUMNS
    )

    prediksi = model.predict(input_df)[0]
    label_hasil = str(prediksi)  # model sudah mengembalikan 'LULUS' / 'TIDAK LULUS' langsung

    # Ambil probabilitas jika model mendukung predict_proba
    proba_text = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        # urutan kolom proba mengikuti model.classes_, bukan diasumsikan [LULUS, TIDAK LULUS]
        kelas = list(model.classes_)
        proba_lulus = proba[kelas.index("LULUS")] * 100 if "LULUS" in kelas else None
        proba_tidak_lulus = proba[kelas.index("TIDAK LULUS")] * 100 if "TIDAK LULUS" in kelas else None
        if proba_lulus is not None and proba_tidak_lulus is not None:
            proba_text = (proba_lulus, proba_tidak_lulus)

    st.divider()
    st.subheader("Hasil Prediksi")

    # st.markdown(f"**Mahasiswa:** {nama_mahasiswa}  \n**Semester:** {semester}")

    if label_hasil == "LULUS":
        st.success(f"✅ Status Prediksi: **{label_hasil}**")
    else:
        st.error(f"❌ Status Prediksi: **{label_hasil}**")

    if proba_text:
        c1, c2 = st.columns(2)
        c1.metric("Probabilitas LULUS", f"{proba_text[0]:.2f}%")
        c2.metric("Probabilitas TIDAK LULUS", f"{proba_text[1]:.2f}%")

    # Tabel detail input mengikuti kolom yang tersedia di form:
    # Semester, KUMULATIF SKS, KUMULATIF IPK, KUMULATIF POINT SKEM
    with st.expander("Lihat detail data input"):
        detail_df = pd.DataFrame([{
            # "Mahasiswa": nama_mahasiswa,
            "Semester": semester,
            "KUMULATIF SKS": kumulatif_sks,
            "KUMULATIF IPK": kumulatif_ipk,
            "KUMULATIF POINT SKEM": kumulatif_skem,
        }])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Catatan: Prediksi ini bersifat estimatif berdasarkan model machine learning "
    "dan tidak menggantikan keputusan resmi akademik. Hanya KUMULATIF SKS, "
    "KUMULATIF IPK, dan KUMULATIF POINT SKEM yang digunakan sebagai fitur prediksi oleh model."
)