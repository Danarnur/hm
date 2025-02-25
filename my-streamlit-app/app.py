import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File {file_path} tidak ditemukan.")
        return None

# Fungsi untuk menampilkan tabel dengan fitur pencarian
def display_searchable_dataframe(df):
    query = st.text_input("Cari data:")
    if query:
        df = df[df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    st.dataframe(df.head(10))  # Tampilkan hanya 10 baris pertama untuk efisiensi

# Sidebar Navigasi
st.sidebar.title("📊 Dashboard Sentimen Analysis")
page = st.sidebar.radio("Pilih Halaman", [
    "🏠 Beranda", 
    "📜 Data Awal", 
    "🏷️ Data Setelah Labeling", 
    "🛠️ Data Setelah Preprocessing", 
    "🧩 Data Binary Relevance", 
    "📉 Training & Validation Loss", 
    "📊 Evaluasi Model"
])

# Halaman Beranda
if page == "🏠 Beranda":
    st.title("📊 Dashboard Analisis Sentimen Multi Aspek")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/NLP.png/600px-NLP.png", use_column_width=True)
    st.write("""
    Aplikasi ini digunakan untuk menganalisis sentimen multi-aspek pada ulasan wisata Telaga Sarangan.
    
    **Fitur utama:**
    - Menampilkan data awal, setelah labeling, preprocessing, dan binary relevance
    - Visualisasi Training & Validation Loss
    - Evaluasi model berdasarkan Precision, Recall, F1-Score, dan Hamming Loss
    """)

# Halaman Data Awal
elif page == "📜 Data Awal":
    st.title("📜 Data Awal Sebelum Labeling")
    data_awal = load_data("data/crawl1.csv")
    if data_awal is not None:
        display_searchable_dataframe(data_awal)

# Halaman Data Setelah Labeling
elif page == "🏷️ Data Setelah Labeling":
    st.title("🏷️ Data Setelah Labeling")
    data_label = load_data("data/data_berlabel.csv")
    if data_label is not None:
        display_searchable_dataframe(data_label)

# Halaman Data Setelah Preprocessing
elif page == "🛠️ Data Setelah Preprocessing":
    st.title("🛠️ Data Setelah Preprocessing")
    data_bersih = load_data("data/data_bersih.csv")
    if data_bersih is not None:
        tahap = st.selectbox("Pilih Tahap Preprocessing", [
            "Cleaning", "Hapus Emoji", "Replace TOM", "Case Folding", "Tokenizing", 
            "Formalisasi", "Stopword Removal", "Stemming"])
        display_searchable_dataframe(data_bersih[[tahap]])

# Halaman Data Binary Relevance
elif page == "🧩 Data Binary Relevance":
    st.title("🧩 Data dalam Bentuk Binary Relevance")
    data_binary = load_data("data/data_label_final.csv")
    if data_binary is not None:
        display_searchable_dataframe(data_binary)

# Halaman Training & Validation Loss
elif page == "📉 Training & Validation Loss":
    st.title("📉 Training & Validation Loss")
    loss_data = load_data("data/loss_data.csv")
    if loss_data is not None:
        epoch_list = loss_data['epoch'].unique()
        selected_epoch = st.selectbox("Pilih Epoch", epoch_list)
        
        epoch_data = loss_data[loss_data['epoch'] == selected_epoch]

        fig, ax = plt.subplots()
        ax.plot(epoch_data['iteration'], epoch_data['train_loss'], label='Training Loss', marker='o')
        ax.plot(epoch_data['iteration'], epoch_data['val_loss'], label='Validation Loss', marker='s')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss pada Epoch {selected_epoch}")
        ax.legend()
        st.pyplot(fig)

# Halaman Evaluasi Model
elif page == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model")
    eval_data = load_data("data/evaluation.csv")
    if eval_data is not None:
        epoch_list = eval_data['epoch'].unique()
        selected_epoch = st.selectbox("Pilih Epoch", epoch_list)

        epoch_eval = eval_data[eval_data['epoch'] == selected_epoch]
        
        st.write(f"**Precision:** {epoch_eval['precision'].values[0]}")
        st.write(f"**Recall:** {epoch_eval['recall'].values[0]}")
        st.write(f"**F1 Score:** {epoch_eval['f1'].values[0]}")
        st.write(f"**Hamming Loss:** {epoch_eval['hamming_loss'].values[0]}")
