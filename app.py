import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----- Pustaka opsional -----
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


# =============================================================================
# KONFIGURASI HALAMAN DAN KONSTANTA
# =============================================================================

st.set_page_config(
    page_title="Dashboard Sentimen & Emosi Publik - Program MBG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_revisi_notebook"
INSET_POSITIVE_PATH = BASE_DIR / "InSet" / "positive.tsv"
INSET_NEGATIVE_PATH = BASE_DIR / "InSet" / "negative.tsv"
NRC_PATH = BASE_DIR / "NRC-EmoLex.txt"
KAMUS_NEGASI_PATH = BASE_DIR / "kamus_negasi.txt"

LABEL_ORDER = ["negatif", "positif"]
LABEL_DISPLAY = {"negatif": "Negatif", "positif": "Positif"}

EMOSI_8 = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
EMOSI_LABEL = {
    "anger": "Marah",
    "anticipation": "Antisipasi",
    "disgust": "Jijik",
    "fear": "Takut",
    "joy": "Senang",
    "sadness": "Sedih",
    "surprise": "Terkejut",
    "trust": "Percaya",
}
EMOSI_COLS = [EMOSI_LABEL[e] for e in EMOSI_8]
RAW_EMOSI_COLS = [f"raw_{EMOSI_LABEL[e]}" for e in EMOSI_8]
PRESENCE_EMOSI_COLS = [f"presence_{EMOSI_LABEL[e]}" for e in EMOSI_8]
WEIGHTED_EMOSI_COLS = [f"weighted_{EMOSI_LABEL[e]}" for e in EMOSI_8]

EMOSI_TIE_PRIORITY = ["anger", "fear", "disgust", "sadness", "joy", "trust", "anticipation", "surprise"]

EMOSI_COLOR = {
    "Marah": "#D62728",
    "Antisipasi": "#FF7F0E",
    "Jijik": "#8C564B",
    "Takut": "#9467BD",
    "Senang": "#2CA02C",
    "Sedih": "#1F77B4",
    "Terkejut": "#E377C2",
    "Percaya": "#17BECF",
    "Tidak Ada": "#BBBBBB",
}

SENTIMEN_COLOR = {"negatif": "#D62728", "positif": "#2CA02C"}

NEGATION_TERMS = {"tidak", "bukan", "jangan", "belum", "kurang", "tanpa", "tak"}

SCENARIO_INFO = {
    "1A": {"Kernel": "Linear", "Kondisi_SMOTE": "Tanpa SMOTE", "Keterangan": "Linear tanpa SMOTE"},
    "1B": {"Kernel": "Linear", "Kondisi_SMOTE": "Dengan SMOTE", "Keterangan": "Linear dengan SMOTE"},
    "2A": {"Kernel": "RBF", "Kondisi_SMOTE": "Tanpa SMOTE", "Keterangan": "RBF tanpa SMOTE"},
    "2B": {"Kernel": "RBF", "Kondisi_SMOTE": "Dengan SMOTE", "Keterangan": "RBF dengan SMOTE"},
    "3A": {"Kernel": "Polynomial", "Kondisi_SMOTE": "Tanpa SMOTE", "Keterangan": "Polynomial tanpa SMOTE"},
    "3B": {"Kernel": "Polynomial", "Kondisi_SMOTE": "Dengan SMOTE", "Keterangan": "Polynomial dengan SMOTE"},
}

SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "g": "tidak", "nggak": "tidak", "ngga": "tidak",
    "tdk": "tidak", "gk": "tidak", "gx": "tidak", "kagak": "tidak", "no": "tidak",
    "bkn": "bukan", "blm": "belum", "belom": "belum", "jgn": "jangan",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "jg": "juga", "sdh": "sudah", "udah": "sudah", "udh": "sudah", "sm": "sama",
    "bgt": "banget", "bngt": "banget", "hrs": "harus", "org": "orang", "tp": "tapi",
    "ttg": "tentang", "spy": "supaya", "dpt": "dapat", "lg": "lagi", "dr": "dari",
    "pd": "pada", "kl": "kalau", "kalo": "kalau", "klu": "kalau", "aja": "saja",
    "aj": "saja", "emg": "memang", "emang": "memang", "gmn": "bagaimana",
    "gimana": "bagaimana", "knp": "kenapa", "mkn": "makan", "msk": "masuk",
    "bener": "benar", "bner": "benar", "bnr": "benar", "skrg": "sekarang",
    "skrng": "sekarang", "skrang": "sekarang", "jd": "jadi", "jdi": "jadi",
    "bs": "bisa", "bsa": "bisa", "sy": "saya", "gw": "saya", "gue": "saya",
    "aku": "saya", "lo": "kamu", "lu": "kamu", "km": "kamu", "loe": "kamu",
    "ttp": "tetap", "ttap": "tetap", "tetp": "tetap", "brp": "berapa",
    "bnyak": "banyak", "bnyk": "banyak", "prgrm": "program", "pgm": "program",
    "mnrt": "menurut", "mnrut": "menurut", "pst": "pasti", "psti": "pasti",
    "denger": "dengar", "dengernya": "dengar", "mulu": "terus",
    "sih": "", "deh": "", "dong": "", "nih": "", "loh": "", "lho": "",
    "wkwk": "", "wkwkwk": "", "haha": "", "hehe": "", "hihi": "", "huhu": "",
}


# =============================================================================
# FUNGSI PRA-PEMROSESAN (sama seperti pipeline pada notebook penelitian)
# =============================================================================

@st.cache_resource
def get_stemmer_and_stopwords():
    """Inisialisasi stemmer Sastrawi dan daftar stopword sesuai pipeline notebook."""
    if not SASTRAWI_AVAILABLE:
        st.toast("Dependensi Sastrawi tidak tersedia. Jalankan environment notebook terlebih dahulu.", icon="⚠️")
        st.error(
            "Pustaka **Sastrawi** tidak tersedia, sehingga pipeline pra-pemrosesan "
            "tidak dapat dijalankan sesuai hasil notebook."
        )
        st.stop()

    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    factory_stop = StopWordRemoverFactory()
    stopword_list = set(factory_stop.get_stop_words())
    return stemmer, stopword_list


def cleansing(text: str) -> str:
    """Langkah 1: Cleansing - hapus URL, mention, hashtag, angka, emoji, tanda baca."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_folding(text: str) -> str:
    """Langkah 2: Case folding - ubah ke huruf kecil."""
    return str(text).lower()


def tokenisasi(text: str) -> list:
    """Langkah 3: Tokenisasi - pecah kalimat menjadi token kata."""
    if not NLTK_AVAILABLE:
        st.toast("Dependensi NLTK tidak tersedia. Jalankan environment notebook terlebih dahulu.", icon="⚠️")
        st.error(
            "Pustaka **NLTK** tidak tersedia, sehingga tokenisasi tidak dapat "
            "dijalankan sesuai pipeline notebook."
        )
        st.stop()
    try:
        return word_tokenize(str(text))
    except LookupError as exc:
        st.toast("Data tokenizer NLTK belum tersedia. Jalankan setup notebook terlebih dahulu.", icon="⚠️")
        st.error(f"Tokenizer NLTK belum siap: `{exc}`")
        st.stop()


def normalisasi_slang(tokens: list, kamus: dict = SLANG_DICT) -> list:
    """Langkah 4: Normalisasi slang - ubah kata tidak baku ke bentuk baku."""
    normalized = []
    for token in tokens:
        replacement = kamus.get(token, token)
        if replacement:
            normalized.extend(str(replacement).split())
    return normalized


def stopword_removal(tokens: list, sw: set) -> list:
    """Langkah 5: Stopword removal - hapus kata umum, KECUALI kata negasi."""
    return [token for token in tokens if token not in sw or token in NEGATION_TERMS]


def stemming_tokens(tokens: list, stemmer) -> list:
    """Langkah 6: Stemming - ubah kata berimbuhan ke bentuk dasar."""
    return [stemmer.stem(token) for token in tokens if str(token).strip()]


def preprocess_pipeline(text: str, stemmer, stopword_list,
                         remove_stopword: bool = True, use_stemming: bool = True) -> dict:
    """Menjalankan seluruh pipeline pra-pemrosesan dan mengembalikan hasil tiap langkah."""
    hasil = {}
    hasil["0_input"] = str(text)
    hasil["1_cleansing"] = cleansing(text)
    hasil["2_case_folding"] = case_folding(hasil["1_cleansing"])
    hasil["3_tokenisasi"] = tokenisasi(hasil["2_case_folding"])
    hasil["4_normalisasi_slang"] = normalisasi_slang(hasil["3_tokenisasi"])
    if remove_stopword:
        hasil["5_stopword_removal"] = stopword_removal(hasil["4_normalisasi_slang"], stopword_list)
    else:
        hasil["5_stopword_removal"] = hasil["4_normalisasi_slang"]
    if use_stemming:
        hasil["6_stemming"] = stemming_tokens(hasil["5_stopword_removal"], stemmer)
    else:
        hasil["6_stemming"] = hasil["5_stopword_removal"]
    hasil["7_clean_text"] = " ".join(hasil["6_stemming"])
    return hasil


def normalize_lexicon_term(term: str, stemmer, stopword_list) -> tuple:
    """Memproses term leksikon dengan pipeline yang sama agar sebanding dengan clean_text."""
    hasil = preprocess_pipeline(term, stemmer, stopword_list, remove_stopword=True, use_stemming=True)
    return tuple(hasil["6_stemming"])


def normalize_manual_negation_term(term: str, stemmer, stopword_list) -> tuple:
    """Memproses frasa negasi manual tanpa stopword removal agar frasa tetap utuh."""
    hasil = preprocess_pipeline(term, stemmer, stopword_list, remove_stopword=False, use_stemming=True)
    return tuple(hasil["6_stemming"])


def tokens_for_phrase_matching(text_or_tokens, stemmer, stopword_list) -> list:
    """Membentuk token untuk pencocokan frasa negasi tanpa menghapus stopword."""
    if isinstance(text_or_tokens, str):
        return preprocess_pipeline(
            text_or_tokens,
            stemmer,
            stopword_list,
            remove_stopword=False,
            use_stemming=True,
        )["6_stemming"]
    return [str(token).strip() for token in text_or_tokens if str(token).strip()]


# =============================================================================
# FUNGSI INSET LEXICON + PENANGANAN NEGASI
# =============================================================================

def read_inset_file(path: Path, default_sign: int) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", header=None, dtype=str, encoding="utf-8", on_bad_lines="skip")
    if raw.shape[1] < 2:
        raw = pd.read_csv(path, sep=";", header=None, dtype=str, encoding="utf-8", on_bad_lines="skip")
    if raw.shape[1] < 2:
        raise ValueError(f"Format file InSet tidak dikenali: {path}")

    df_lex = raw.iloc[:, :2].copy()
    df_lex.columns = ["term", "score"]
    df_lex["score"] = pd.to_numeric(df_lex["score"], errors="coerce")
    df_lex = df_lex.dropna(subset=["term"])
    if df_lex["score"].isna().any():
        df_lex = df_lex.dropna(subset=["score"])
    if df_lex.empty:
        df_lex = raw.iloc[:, [0]].copy()
        df_lex.columns = ["term"]
        df_lex["score"] = default_sign
    df_lex["score"] = df_lex["score"].astype(float)
    return df_lex


@st.cache_resource
def load_inset_lexicon():
    """Memuat InSet Lexicon (positive.tsv + negative.tsv) menjadi dictionary skor."""
    if not (INSET_POSITIVE_PATH.exists() and INSET_NEGATIVE_PATH.exists()):
        st.toast("File leksikon InSet tidak tersedia. Jalankan notebook terlebih dahulu.", icon="⚠️")
        st.error(
            f"File InSet wajib tersedia: `{INSET_POSITIVE_PATH}` dan "
            f"`{INSET_NEGATIVE_PATH}`."
        )
        st.stop()

    stemmer, stopword_list = get_stemmer_and_stopwords()
    pos_df = read_inset_file(INSET_POSITIVE_PATH, default_sign=1)
    neg_df = read_inset_file(INSET_NEGATIVE_PATH, default_sign=-1)
    lex_df = pd.concat([pos_df, neg_df], ignore_index=True)

    inset_dict = {}
    for _, row in lex_df.iterrows():
        term = str(row["term"]).strip()
        score = float(row["score"])
        key = normalize_lexicon_term(term, stemmer, stopword_list)
        if not key:
            continue
        if key not in inset_dict or abs(score) > abs(inset_dict[key]):
            inset_dict[key] = score

    max_ngram = max((len(key) for key in inset_dict.keys()), default=1)
    return inset_dict, max_ngram


def has_sentiment_match_at(tokens, idx, inset_dict, max_ngram):
    max_len = min(max_ngram, len(tokens) - idx)
    for n in range(max_len, 0, -1):
        if tuple(tokens[idx:idx + n]) in inset_dict:
            return True
    return False


@st.cache_data(show_spinner=False)
def load_kamus_negasi():
    """Memuat kamus negasi manual format notebook: frasa_negasi + skor_manual."""
    kamus = {}
    if not KAMUS_NEGASI_PATH.exists():
        st.toast("File kamus negasi tidak tersedia. Jalankan notebook terlebih dahulu.", icon="⚠️")
        st.error(f"File wajib `{KAMUS_NEGASI_PATH}` tidak ditemukan.")
        st.stop()

    stemmer, stopword_list = get_stemmer_and_stopwords()
    try:
        neg_df = pd.read_csv(KAMUS_NEGASI_PATH, sep="\t", dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        neg_df = pd.read_csv(KAMUS_NEGASI_PATH, sep="\t", dtype=str, encoding="latin-1")

    required_cols = {"frasa_negasi", "padanan_makna", "skor_manual", "kategori"}
    if not required_cols.issubset(set(neg_df.columns)):
        missing_cols = sorted(required_cols - set(neg_df.columns))
        st.toast("Format kamus negasi tidak sesuai output notebook.", icon="⚠️")
        st.error(f"Kolom wajib pada `kamus_negasi.txt` tidak ada: {missing_cols}")
        st.stop()

    neg_df["skor_manual"] = pd.to_numeric(neg_df["skor_manual"], errors="coerce")
    neg_df = neg_df.dropna(subset=["frasa_negasi", "skor_manual"])

    for _, row in neg_df.iterrows():
        key = normalize_manual_negation_term(row["frasa_negasi"], stemmer, stopword_list)
        if len(key) < 2:
            continue
        kamus[key] = {
            "frasa_negasi": str(row["frasa_negasi"]).strip(),
            "padanan_makna": str(row.get("padanan_makna", "")).strip(),
            "skor_manual": float(row["skor_manual"]),
            "kategori": str(row.get("kategori", "")).strip(),
            "keterangan": str(row.get("keterangan", "")).strip(),
        }

    max_ngram = max((len(k) for k in kamus.keys()), default=1) if kamus else 1
    return kamus, max_ngram, neg_df

def score_by_inset_with_negation(text_or_tokens, inset_dict, max_ngram):
    """
    Menghitung skor sentimen InSet dengan negasi manual berbasis kamus frasa.
    
    Frasa pada kamus_negasi.txt dicocokkan lebih dulu. Jika cocok, skor_manual
    dipakai langsung dan token penyusun frasa tidak dinilai ulang sebagai InSet biasa.
    """
    kamus_negasi, neg_max_ngram, _ = load_kamus_negasi()
    stemmer, stopword_list = get_stemmer_and_stopwords()

    tokens = tokens_for_phrase_matching(text_or_tokens, stemmer, stopword_list)

    total_score = 0.0
    matches = []

    i = 0
    while i < len(tokens):
        found = False
        
        # 1. Cek kamus negasi manual.
        max_neg_len = min(neg_max_ngram, len(tokens) - i)
        for n in range(max_neg_len, 1, -1):
            key = tuple(tokens[i:i + n])
            if key in kamus_negasi:
                info = kamus_negasi[key]
                final_score = float(info["skor_manual"])
                total_score += final_score
                matches.append({
                    "term": " ".join(key),
                    "posisi": i,
                    "jenis_match": "negasi_manual",
                    "frasa_negasi_asli": info["frasa_negasi"],
                    "padanan_makna": info["padanan_makna"],
                    "kategori": info["kategori"],
                    "skor_asli": np.nan,
                    "skor_akhir": final_score,
                    "keterangan": info["keterangan"],
                })
                i += n
                found = True
                break
                
        if found:
            continue
            
        # 2. Cek InSet biasa jika tidak ada frasa negasi manual.
        max_len = min(max_ngram, len(tokens) - i)
        for n in range(max_len, 0, -1):
            key = tuple(tokens[i:i + n])
            if key in inset_dict:
                base_score = float(inset_dict[key])
                final_score = base_score
                total_score += final_score
                matches.append({
                    "term": " ".join(key),
                    "posisi": i,
                    "jenis_match": "inset_biasa",
                    "frasa_negasi_asli": "",
                    "padanan_makna": "",
                    "kategori": "positif" if base_score > 0 else "negatif",
                    "skor_asli": base_score,
                    "skor_akhir": final_score,
                    "keterangan": "Term InSet tanpa aturan negasi manual",
                })
                i += n
                found = True
                break
        if not found:
            i += 1

    return total_score, matches


def tentukan_label(score: float) -> str:
    return "positif" if score > 0 else "negatif"


# =============================================================================
# FUNGSI NRC EMOTION LEXICON
# =============================================================================

def normalize_colname(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def is_active_value(value) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def choose_indonesian_term_column(columns):
    columns = list(columns)
    priority = ["indonesian_word", "indonesian", "indonesian_translation",
                "translation", "kata_indonesia", "kata", "term_indonesia", "term"]
    for candidate in priority:
        if candidate in columns:
            return candidate
    for candidate in ["word", "english_word"]:
        if candidate in columns:
            return candidate
    return columns[0]


@st.cache_resource
def load_nrc_lexicon():
    """Memuat NRC Emotion Lexicon versi Bahasa Indonesia menjadi dictionary emosi."""
    if not NRC_PATH.exists():
        st.toast("File NRC Emotion Lexicon tidak tersedia. Jalankan notebook terlebih dahulu.", icon="⚠️")
        st.error(f"File wajib `{NRC_PATH}` tidak ditemukan.")
        st.stop()

    stemmer, stopword_list = get_stemmer_and_stopwords()

    try:
        df_nrc = pd.read_csv(NRC_PATH, sep="\t", dtype=str, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df_nrc = pd.read_csv(NRC_PATH, sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip")

    df_nrc = df_nrc.rename(columns={col: normalize_colname(col) for col in df_nrc.columns})
    nrc_dict = defaultdict(set)
    available_emotions = [e for e in EMOSI_8 if e in df_nrc.columns]

    if len(available_emotions) >= 4:
        non_emotion_cols = [c for c in df_nrc.columns if c not in set(EMOSI_8) | {"positive", "negative"}]
        term_col = choose_indonesian_term_column(non_emotion_cols)
        for _, row in df_nrc.iterrows():
            key = normalize_lexicon_term(row.get(term_col, ""), stemmer, stopword_list)
            if not key:
                continue
            for emotion in available_emotions:
                if is_active_value(row.get(emotion, 0)):
                    nrc_dict[key].add(emotion)
    elif {"emotion", "association"}.issubset(set(df_nrc.columns)):
        term_candidates = [c for c in df_nrc.columns if c not in {"emotion", "association"}]
        term_col = choose_indonesian_term_column(term_candidates)
        for _, row in df_nrc.iterrows():
            emotion = normalize_colname(row.get("emotion", ""))
            if emotion in EMOSI_8 and is_active_value(row.get("association", 0)):
                key = normalize_lexicon_term(row.get(term_col, ""), stemmer, stopword_list)
                if key:
                    nrc_dict[key].add(emotion)
    else:
        raw = pd.read_csv(NRC_PATH, sep="\t", header=None, dtype=str, encoding="utf-8", on_bad_lines="skip")
        if raw.shape[1] >= 12:
            term_col_idx = raw.shape[1] - 1
            emotion_position = {"anger": 1, "anticipation": 2, "disgust": 3, "fear": 4, "joy": 5,
                                 "sadness": 8, "surprise": 9, "trust": 10}
            for _, row in raw.iterrows():
                key = normalize_lexicon_term(row.iloc[term_col_idx], stemmer, stopword_list)
                if not key:
                    continue
                for emotion, pos in emotion_position.items():
                    if pos < len(row) and is_active_value(row.iloc[pos]):
                        nrc_dict[key].add(emotion)
        elif raw.shape[1] == 3:
            raw.columns = ["term", "emotion", "association"]
            for _, row in raw.iterrows():
                emotion = normalize_colname(row.get("emotion", ""))
                if emotion in EMOSI_8 and is_active_value(row.get("association", 0)):
                    key = normalize_lexicon_term(row.get("term", ""), stemmer, stopword_list)
                    if key:
                        nrc_dict[key].add(emotion)

    nrc_dict = {key: set(value) for key, value in nrc_dict.items() if value}

    kamus_negasi, _, _ = load_kamus_negasi()
    for neg_key, neg_info in kamus_negasi.items():
        if len(neg_key) < 2:
            continue

        phrase_emotions = set(nrc_dict.get(neg_key, set()))
        padanan_key = normalize_lexicon_term(neg_info.get("padanan_makna", ""), stemmer, stopword_list)
        if padanan_key:
            phrase_emotions.update(nrc_dict.get(padanan_key, set()))

        if not phrase_emotions:
            for token in neg_key:
                phrase_emotions.update(nrc_dict.get((token,), set()))

        if phrase_emotions:
            nrc_dict[neg_key] = phrase_emotions

    max_ngram = max((len(key) for key in nrc_dict.keys()), default=1)
    return nrc_dict, max_ngram


def score_nrc_per_comment(text_or_tokens, nrc_dict, max_ngram):
    """Menghitung emosi NRC per komentar: 1 komentar = 1 emosi dominan final."""
    stemmer, stopword_list = get_stemmer_and_stopwords()
    tokens = tokens_for_phrase_matching(text_or_tokens, stemmer, stopword_list)
    kamus_negasi, _, _ = load_kamus_negasi()
    raw_counts = {emotion: 0 for emotion in EMOSI_8}
    weighted_scores = {emotion: 0.0 for emotion in EMOSI_8}
    matches = []

    i = 0
    while i < len(tokens):
        found = False
        max_len = min(max_ngram, len(tokens) - i)
        for n in range(max_len, 0, -1):
            key = tuple(tokens[i:i + n])
            if key in nrc_dict:
                emotions = sorted(nrc_dict[key])
                if not emotions:
                    break
                term_weight = 1 / len(emotions)
                for emotion in emotions:
                    raw_counts[emotion] += 1
                    weighted_scores[emotion] += term_weight
                matches.append({
                    "term": " ".join(key),
                    "jenis_match": "frasa_negasi_manual" if key in kamus_negasi else "term_nrc",
                    "emosi_kode": emotions,
                    "emosi": ", ".join(EMOSI_LABEL[e] for e in emotions),
                    "bobot_per_emosi": round(term_weight, 4),
                    "posisi": i,
                })
                i += n
                found = True
                break
        if not found:
            i += 1

    presence = {emotion: int(raw_counts[emotion] > 0) for emotion in EMOSI_8}
    detected = [EMOSI_LABEL[e] for e in EMOSI_8 if presence[e] == 1]

    if sum(raw_counts.values()) == 0:
        dominant_label = "Tidak Ada"
        dominant_score = 0.0
        dominant_terms = []
        metode_pemilihan = "Tidak ada term NRC yang cocok"
    else:
        max_score = max(weighted_scores.values())
        candidate_codes = [e for e in EMOSI_8 if np.isclose(weighted_scores[e], max_score) and max_score > 0]
        if len(candidate_codes) == 1:
            dominant_code = candidate_codes[0]
            metode_pemilihan = "Skor tertimbang tertinggi"
        else:
            dominant_code = next(e for e in EMOSI_TIE_PRIORITY if e in candidate_codes)
            metode_pemilihan = "Tie-break karena skor tertimbang seri"

        dominant_label = EMOSI_LABEL[dominant_code]
        dominant_score = weighted_scores[dominant_code]
        dominant_terms = []
        for match in matches:
            if dominant_code in match["emosi_kode"]:
                dominant_terms.append(match["term"])
        dominant_terms = list(dict.fromkeys(dominant_terms))

    result = {
        "raw_counts": raw_counts,
        "presence": presence,
        "weighted_scores": weighted_scores,
        "jumlah_emosi_terdeteksi": sum(presence.values()),
        "emosi_terdeteksi": ", ".join(detected) if detected else "Tidak Ada",
        "emosi_dominan": dominant_label,
        "skor_emosi_dominan": round(dominant_score, 4),
        "bukti_emosi_dominan": ", ".join(dominant_terms) if dominant_terms else "-",
        "metode_pemilihan_emosi": metode_pemilihan,
        "matches": matches,
    }
    return result


# =============================================================================
# FUNGSI PEMUATAN DATA HASIL PENELITIAN
# =============================================================================

REQUIRED_OUTPUT_FILES = {
    "Data lengkap sentimen dan emosi": OUTPUT_DIR / "hasil_sentimen_emosi_mbg_revisi.csv",
    "Ringkasan evaluasi SVM": OUTPUT_DIR / "ringkasan_evaluasi_skenario_1a_3b.csv",
    "Pengaruh SMOTE per kernel": OUTPUT_DIR / "pengaruh_smote_per_kernel.csv",
    "Distribusi SMOTE": OUTPUT_DIR / "distribusi_smote.csv",
    "Ringkasan NRC": OUTPUT_DIR / "ringkasan_nrc_emosi_dominan_raw_presence.csv",
    "Tabulasi emosi per sentimen": OUTPUT_DIR / "emosi_dominan_per_sentimen.csv",
    "Metadata notebook": OUTPUT_DIR / "metadata_revisi_notebook.json",
    "TF-IDF vectorizer": OUTPUT_DIR / "tfidf_vectorizer.joblib",
}


def show_missing_notebook_outputs(missing_files):
    st.toast("File output notebook belum lengkap. Jalankan notebook terlebih dahulu.", icon="⚠️")
    st.error(
        "Aplikasi Streamlit ini hanya memakai artefak hasil ekspor notebook. "
        "Tidak ada data pengganti yang digunakan."
    )
    st.markdown("**File yang belum tersedia:**")
    st.write([str(path.relative_to(BASE_DIR)) for path in missing_files])
    st.stop()


def validate_required_outputs():
    missing = [path for path in REQUIRED_OUTPUT_FILES.values() if not path.exists()]
    if missing:
        show_missing_notebook_outputs(missing)


@st.cache_data
def load_main_data():
    path = OUTPUT_DIR / "hasil_sentimen_emosi_mbg_revisi.csv"
    return pd.read_csv(path)


@st.cache_data
def load_results_df():
    path = OUTPUT_DIR / "ringkasan_evaluasi_skenario_1a_3b.csv"
    return pd.read_csv(path)


@st.cache_data
def load_smote_effect():
    path = OUTPUT_DIR / "pengaruh_smote_per_kernel.csv"
    return pd.read_csv(path)


@st.cache_data
def load_smote_distribution():
    path = OUTPUT_DIR / "distribusi_smote.csv"
    df_smote = pd.read_csv(path)
    first_col = df_smote.columns[0]
    if first_col != "Kelas":
        df_smote = df_smote.rename(columns={first_col: "Kelas"})
    return df_smote


@st.cache_data
def load_nrc_summary():
    path = OUTPUT_DIR / "ringkasan_nrc_emosi_dominan_raw_presence.csv"
    return pd.read_csv(path)


@st.cache_data
def load_emosi_per_sentimen():
    path = OUTPUT_DIR / "emosi_dominan_per_sentimen.csv"
    return pd.read_csv(path).set_index("sentimen_pred")


@st.cache_data
def load_metadata():
    path = OUTPUT_DIR / "metadata_revisi_notebook.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_svm_model(scenario_id):
    """Memuat model SVM dan TF-IDF vectorizer hasil ekspor notebook."""
    try:
        import joblib
    except ImportError as exc:
        st.toast("Dependensi joblib tidak tersedia. Jalankan environment notebook terlebih dahulu.", icon="⚠️")
        st.error(f"Pustaka **joblib** tidak tersedia: `{exc}`")
        st.stop()

    model_path = OUTPUT_DIR / f"model_svm_terpilih_{scenario_id}.joblib"
    tfidf_path = OUTPUT_DIR / "tfidf_vectorizer.joblib"
    missing = [path for path in [model_path, tfidf_path] if not path.exists()]
    if missing:
        show_missing_notebook_outputs(missing)

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf


CONTOH_PREPROCESSING = [
    {
        "label": "Contoh 1 - Komentar Singkat",
        "teks": "@tempodotco 36 Trilyun buat THR Guru",
    },
    {
        "label": "Contoh 2 - Komentar dengan Slang",
        "teks": "@kompascom Tolong, jgn sering dikasih pidato mulu. Pusing rakyat dengernya!",
    },
    {
        "label": "Contoh 3 - Komentar Panjang dengan URL",
        "teks": "@kompascom Bukan gosip tapi fakta, Proyek MBG itu Modus Korupsi berkedok efisiensi anggaran https://t.co/FZeowExenZ",
    },
    {
        "label": "Contoh 4 - Kalimat dengan Negasi",
        "teks": "Program MBG ini tidak jelek, bahkan sangat membantu anak-anak sekolah, bukan pencitraan semata.",
    },
]


# =============================================================================
# FUNGSI BANTUAN VISUALISASI
# =============================================================================

def format_pct(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Negatif", "Positif"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Label Prediksi")
    ax.set_ylabel("Label Aktual")
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_bar_metrics(results_df, title):
    plot_df = results_df.set_index("Skenario")[["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Nilai Metrik")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Skenario")
    ax.legend(["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"],
              loc="lower right", fontsize=8)
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_smote_effect(smote_effect_df):
    x = np.arange(len(smote_effect_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, smote_effect_df["F1_Tanpa_SMOTE"], width, label="Tanpa SMOTE", color="#9ecae1")
    ax.bar(x + width / 2, smote_effect_df["F1_Dengan_SMOTE"], width, label="Dengan SMOTE", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels(smote_effect_df["Kernel"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score Makro")
    ax.set_title("Pengaruh SMOTE terhadap F1-Score Makro per Kernel", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_emosi_dominan(nrc_summary_df):
    plot_data = nrc_summary_df.sort_values("Jumlah_Komentar_Sebagai_Emosi_Dominan", ascending=True)
    colors = [EMOSI_COLOR.get(e, "#999999") for e in plot_data["Emosi"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot_data["Emosi"], plot_data["Jumlah_Komentar_Sebagai_Emosi_Dominan"], color=colors)
    ax.set_xlabel("Jumlah Komentar")
    ax.set_title("Distribusi Emosi Dominan NRC per Komentar", fontweight="bold")
    for i, (val, name) in enumerate(zip(plot_data["Jumlah_Komentar_Sebagai_Emosi_Dominan"], plot_data["Emosi"])):
        ax.text(val + 3, i, str(val), va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_emosi_per_sentimen(emosi_per_sentimen_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, sentimen in zip(axes, LABEL_ORDER):
        if sentimen not in emosi_per_sentimen_df.index:
            continue
        data_plot = emosi_per_sentimen_df.loc[sentimen]
        data_plot = data_plot.drop(labels=["Tidak Ada"], errors="ignore").sort_values(ascending=True)
        colors = [EMOSI_COLOR.get(e, "#999999") for e in data_plot.index]
        ax.barh(data_plot.index, data_plot.values, color=colors)
        ax.set_title(f"Emosi Dominan - Sentimen {LABEL_DISPLAY.get(sentimen, sentimen).capitalize()}", fontweight="bold")
        ax.set_xlabel("Jumlah Komentar")
    fig.tight_layout()
    return fig


def plot_smote_distribution(smote_dist_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(smote_dist_df))
    width = 0.35
    ax.bar(x - width / 2, smote_dist_df["Sebelum_SMOTE"], width, label="Sebelum SMOTE", color="#fdae6b")
    ax.bar(x + width / 2, smote_dist_df["Sesudah_SMOTE"], width, label="Sesudah SMOTE", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_DISPLAY.get(k, k) for k in smote_dist_df["Kelas"]])
    ax.set_ylabel("Jumlah Data")
    ax.set_title("Distribusi Kelas Sebelum dan Sesudah SMOTE (Data Latih)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_wordcloud(text, title):
    if not WORDCLOUD_AVAILABLE:
        st.info("Pustaka 'wordcloud' belum terinstal. Jalankan: pip install wordcloud")
        return None
    if not text or not text.strip():
        st.info(f"Tidak ada teks yang tersedia untuk word cloud {title}.")
        return None
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_emosi_bar_single(weighted_scores, dominant_label):
    """Bar chart skor tertimbang 8 emosi untuk satu komentar, highlight emosi dominan."""
    labels = [EMOSI_LABEL[e] for e in EMOSI_8]
    values = [weighted_scores[e] for e in EMOSI_8]
    colors = []
    for label in labels:
        if label == dominant_label:
            colors.append(EMOSI_COLOR.get(label, "#333333"))
        else:
            colors.append("#D9D9D9")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Skor Tertimbang")
    ax.set_title("Skor Tertimbang 8 Emosi NRC per Komentar", fontweight="bold")
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🍽️ Dashboard MBG")
st.sidebar.markdown(
    "**Analisis Sentimen & Emosi Publik**\n\n"
    "Program Makan Bergizi Gratis (MBG)\n\n"
    "Sumber data: 1.286 reply berbahasa Indonesia dari 4 postingan viral "
    "di platform X (@kompascom, @CNNIndonesia, @tempodotco, @Ghurem2), "
    "periode 21 Januari - 27 Februari 2026."
)

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Ringkasan Umum",
        "Perbandingan Model SVM",
        "Pengaruh SMOTE",
        "Peta Emosi Publik",
        "Word Cloud",
        "Eksplorasi Data",
        "Pra-pemrosesan Data",
        "Pengujian Interaktif",
    ],
)

st.sidebar.markdown("---")

validate_required_outputs()

df = load_main_data()
results_df = load_results_df()
smote_effect_df = load_smote_effect()
smote_dist_df = load_smote_distribution()
nrc_summary_df = load_nrc_summary()
emosi_per_sentimen_df = load_emosi_per_sentimen()
metadata = load_metadata()

selected_scenario = metadata.get("selected_scenario")
if not selected_scenario:
    st.toast("Metadata notebook belum memuat selected_scenario. Jalankan notebook terlebih dahulu.", icon="⚠️")
    st.error("File metadata tidak memuat `selected_scenario`.")
    st.stop()

best_smote_scenario = metadata.get("best_smote_scenario", selected_scenario)
best_overall_scenario = metadata.get("best_overall_scenario", selected_scenario)
model_output_path = OUTPUT_DIR / f"model_svm_terpilih_{selected_scenario}.joblib"
if not model_output_path.exists():
    show_missing_notebook_outputs([model_output_path])


# =============================================================================
# HALAMAN 1: RINGKASAN UMUM
# =============================================================================

if page == "Ringkasan Umum":
    st.title("Dashboard Analisis Sentimen dan Emosi Publik")
    st.subheader("Program Makan Bergizi Gratis (MBG)")
    st.caption(
        "Skripsi: Analisis Sentimen dan Emosi Publik terhadap Program Makan "
        "Bergizi Gratis Menggunakan Support Vector Machine dan NRC Emotion Lexicon"
    )
    st.markdown("---")

    total_data = len(df)
    jml_negatif = int((df["sentimen_pred"] == "negatif").sum())
    jml_positif = int((df["sentimen_pred"] == "positif").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Komentar Bersih", f"{total_data:,}")
    col2.metric("Sentimen Negatif", f"{jml_negatif:,}", f"{jml_negatif/total_data*100:.1f}%")
    col3.metric("Sentimen Positif", f"{jml_positif:,}", f"{jml_positif/total_data*100:.1f}%")

    best_row = results_df.sort_values("F1_Macro", ascending=False).iloc[0]
    col4.metric(
        f"Model Terbaik ({best_row['Skenario']})",
        f"Akurasi {best_row['Accuracy']*100:.2f}%",
        f"F1-Macro {best_row['F1_Macro']*100:.2f}%",
    )

    st.markdown("---")

    st.subheader("Distribusi Sentimen: Label Awal (InSet Lexicon) vs Prediksi Model SVM")
    col_a, col_b = st.columns(2)

    with col_a:
        label_awal = df["sentimen"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.bar([LABEL_DISPLAY[k] for k in label_awal.index], label_awal.values,
               color=[SENTIMEN_COLOR[k] for k in label_awal.index])
        ax.set_title("Label Awal (InSet Lexicon)", fontweight="bold")
        ax.set_ylabel("Jumlah Komentar")
        for i, v in enumerate(label_awal.values):
            ax.text(i, v + 5, str(int(v)), ha="center", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        total_awal = label_awal.sum()
        st.caption(
            f"Negatif: {label_awal['negatif']:,} ({label_awal['negatif']/total_awal*100:.2f}%) | "
            f"Positif: {label_awal['positif']:,} ({label_awal['positif']/total_awal*100:.2f}%)"
        )

    with col_b:
        label_pred = df["sentimen_pred"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.bar([LABEL_DISPLAY[k] for k in label_pred.index], label_pred.values,
               color=[SENTIMEN_COLOR[k] for k in label_pred.index])
        ax.set_title(f"Prediksi Model Terbaik ({selected_scenario})", fontweight="bold")
        ax.set_ylabel("Jumlah Komentar")
        for i, v in enumerate(label_pred.values):
            ax.text(i, v + 5, str(int(v)), ha="center", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        total_pred = label_pred.sum()
        st.caption(
            f"Negatif: {label_pred['negatif']:,} ({label_pred['negatif']/total_pred*100:.2f}%) | "
            f"Positif: {label_pred['positif']:,} ({label_pred['positif']/total_pred*100:.2f}%)"
        )

    st.markdown("---")

    st.subheader("Ringkasan Model SVM Terbaik")
    info = SCENARIO_INFO.get(selected_scenario, SCENARIO_INFO["2B"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Skenario", selected_scenario)
    c2.metric("Kernel", info["Kernel"])
    c3.metric("Kondisi Data Latih", info["Kondisi_SMOTE"])
    c4.metric("Accuracy", format_pct(best_row["Accuracy"]))
    c5.metric("F1-Score Makro", format_pct(best_row["F1_Macro"]))

    st.info(
        f"Model terbaik dengan SMOTE: **Skenario {best_smote_scenario}**. "
        f"Model terbaik keseluruhan (semua skenario): **Skenario {best_overall_scenario}**. "
        f"Model yang digunakan untuk memprediksi seluruh dataset: **Skenario {selected_scenario}**."
    )

    st.markdown("---")

    st.subheader("Ringkasan Sumber Dataset")
    st.dataframe(
        pd.DataFrame({
            "File": ["MBG1.csv", "MBG2.csv", "MBG3.csv", "MBG4.csv", "Total"],
            "Akun Sumber": ["@kompascom", "@CNNIndonesia", "@tempodotco", "@Ghurem2", "4 akun"],
            "Tanggal Postingan": ["13 Feb 2026", "13 Feb 2026", "21 Jan 2026", "16 Feb 2026", "Jan-Feb 2026"],
            "Jumlah Reply Mentah": [352, 367, 349, 367, 1435],
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Dari 1.435 reply mentah, 1.294 (90,2%) terdeteksi berbahasa Indonesia, "
        "dan setelah pra-pemrosesan menghasilkan 1.286 data bersih (8 data "
        "menghasilkan teks kosong dan dihapus)."
    )


# =============================================================================
# HALAMAN 2: PERBANDINGAN MODEL SVM
# =============================================================================

elif page == "Perbandingan Model SVM":
    st.title("Perbandingan Kinerja Model SVM")
    st.markdown(
        "Perbandingan enam skenario eksperimen: tiga kernel SVM (Linear, RBF, "
        "Polynomial), masing-masing pada kondisi **tanpa SMOTE** dan **dengan SMOTE**, "
        "menggunakan parameter terbaik hasil GridSearchCV (5-fold cross validation)."
    )
    st.markdown("---")

    st.subheader("Daftar Skenario Pengujian")
    scenario_table = pd.DataFrame([{"Skenario": sid, **info} for sid, info in SCENARIO_INFO.items()])
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Tabel Hasil Evaluasi Skenario 1A - 3B")
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro",
                "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(2)

    st.dataframe(
        display_df[["Skenario", "Kernel", "Kondisi_SMOTE", "Accuracy",
                     "Precision_Macro", "Recall_Macro", "F1_Macro", "F1_Weighted"]].rename(columns={
                         "Accuracy": "Accuracy (%)",
                         "Precision_Macro": "Precision Macro (%)",
                         "Recall_Macro": "Recall Macro (%)",
                         "F1_Macro": "F1-Macro (%)",
                         "F1_Weighted": "F1-Weighted (%)",
                     }),
        use_container_width=True,
        hide_index=True,
    )

    best_row = results_df.sort_values("F1_Macro", ascending=False).iloc[0]
    st.success(
        f"Model dengan F1-Score Makro tertinggi: **Skenario {best_row['Skenario']} "
        f"({best_row['Kernel']}, {best_row['Kondisi_SMOTE']})** dengan "
        f"Accuracy = {format_pct(best_row['Accuracy'])} dan "
        f"F1-Macro = {format_pct(best_row['F1_Macro'])}."
    )

    st.markdown("---")

    st.subheader("Grafik Perbandingan Metrik Evaluasi")
    fig = plot_bar_metrics(results_df, "Perbandingan Metrik Evaluasi Skenario 1A - 3B")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Detail per Skenario dari File Evaluasi Notebook")
    scenario_choice = st.selectbox(
        "Pilih skenario untuk melihat detail",
        options=list(SCENARIO_INFO.keys()),
        format_func=lambda s: f"{s} - {SCENARIO_INFO[s]['Keterangan']}",
        index=list(SCENARIO_INFO.keys()).index(selected_scenario) if selected_scenario in SCENARIO_INFO else 0,
    )

    scenario_row = results_df.loc[results_df["Skenario"] == scenario_choice].iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", format_pct(scenario_row["Accuracy"]))
    metric_cols[1].metric("Precision Macro", format_pct(scenario_row["Precision_Macro"]))
    metric_cols[2].metric("Recall Macro", format_pct(scenario_row["Recall_Macro"]))
    metric_cols[3].metric("F1-Macro", format_pct(scenario_row["F1_Macro"]))
    st.dataframe(
        pd.DataFrame([scenario_row]).rename(columns={
            "Best_Params": "Parameter Terbaik",
            "Best_CV_F1_Macro": "Best CV F1-Macro",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Classification report per kelas dan confusion matrix tidak ditampilkan "
        "karena file tersebut tidak diekspor oleh notebook. Aplikasi tidak memakai "
        "angka hardcoded sebagai pengganti."
    )


# =============================================================================
# HALAMAN 3: PENGARUH SMOTE
# =============================================================================

elif page == "Pengaruh SMOTE":
    st.title("Pengaruh SMOTE terhadap Kinerja SVM")
    st.markdown(
        "Halaman ini membandingkan kinerja model SVM sebelum dan setelah "
        "penerapan **SMOTE (Synthetic Minority Over-sampling Technique)** "
        "pada data latih, untuk masing-masing kernel."
    )
    st.markdown("---")

    st.subheader("Distribusi Kelas pada Data Latih Sebelum dan Sesudah SMOTE")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = plot_smote_distribution(smote_dist_df)
        st.pyplot(fig)
    with col2:
        st.dataframe(
            smote_dist_df.assign(Kelas=lambda d: d["Kelas"].map(LABEL_DISPLAY)),
            use_container_width=True,
            hide_index=True,
        )
        sebelum_negatif = int(smote_dist_df.loc[smote_dist_df["Kelas"] == "negatif", "Sebelum_SMOTE"].iloc[0])
        sebelum_positif = int(smote_dist_df.loc[smote_dist_df["Kelas"] == "positif", "Sebelum_SMOTE"].iloc[0])
        sesudah_negatif = int(smote_dist_df.loc[smote_dist_df["Kelas"] == "negatif", "Sesudah_SMOTE"].iloc[0])
        sesudah_positif = int(smote_dist_df.loc[smote_dist_df["Kelas"] == "positif", "Sesudah_SMOTE"].iloc[0])
        st.markdown(
            f"- **Sebelum SMOTE**: {sebelum_negatif:,} data negatif, "
            f"{sebelum_positif:,} data positif "
            f"(total {sebelum_negatif + sebelum_positif:,})\n"
            f"- **Sesudah SMOTE**: {sesudah_negatif:,} data negatif, "
            f"{sesudah_positif:,} data positif "
            f"(total {sesudah_negatif + sesudah_positif:,})\n"
            "- SMOTE diterapkan **hanya pada data latih** dengan `k_neighbors=5`; "
            "data uji (20%) tidak mengalami SMOTE."
        )

    st.markdown("---")

    st.subheader("Perbandingan F1-Score Makro dan Accuracy per Kernel")
    display_smote = smote_effect_df.copy()
    for col in ["F1_Tanpa_SMOTE", "F1_Dengan_SMOTE", "Delta_F1", "Accuracy_Tanpa_SMOTE",
                "Accuracy_Dengan_SMOTE", "Delta_Accuracy"]:
        if col in display_smote.columns:
            display_smote[col] = (display_smote[col] * 100).round(2)

    st.dataframe(
        display_smote.rename(columns={
            "F1_Tanpa_SMOTE": "F1-Macro Tanpa SMOTE (%)",
            "F1_Dengan_SMOTE": "F1-Macro Dengan SMOTE (%)",
            "Delta_F1": "Selisih F1-Macro (poin %)",
            "Accuracy_Tanpa_SMOTE": "Accuracy Tanpa SMOTE (%)",
            "Accuracy_Dengan_SMOTE": "Accuracy Dengan SMOTE (%)",
            "Delta_Accuracy": "Selisih Accuracy (poin %)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader("Grafik Pengaruh SMOTE terhadap F1-Score Makro")
    fig = plot_smote_effect(smote_effect_df)
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Interpretasi")
    st.markdown(
        "Berdasarkan kriteria pada rancangan penelitian, peningkatan F1-Score "
        "makro lebih dari **5 poin persentase** dianggap sebagai bukti kontribusi "
        "nyata SMOTE. Pada penelitian ini, seluruh selisih F1-Score makro "
        "berkisar antara **-0,61 hingga +0,80 poin persentase**, jauh di bawah "
        "ambang batas tersebut."
    )
    col1, col2, col3 = st.columns(3)
    for col, row in zip([col1, col2, col3], smote_effect_df.itertuples()):
        delta_f1 = row.Delta_F1 * 100
        arrow = "🔼" if delta_f1 > 0 else "🔽"
        col.metric(f"{row.Kernel}", f"{arrow} {delta_f1:+.2f} poin F1-Macro",
                   f"{row.Skenario_Tanpa_SMOTE} → {row.Skenario_Dengan_SMOTE}")

    st.info(
        "**Kesimpulan**: SMOTE tidak memberikan pengaruh signifikan terhadap "
        "metrik agregat (accuracy & F1-Macro), namun tetap membantu meningkatkan "
        "**recall kelas Positif (kelas minoritas)** pada ketiga kernel, sehingga "
        "model menjadi sedikit lebih sensitif dalam mengenali komentar bersentimen positif."
    )


# =============================================================================
# HALAMAN 4: PETA EMOSI PUBLIK
# =============================================================================

elif page == "Peta Emosi Publik":
    st.title("Peta Emosi Publik terhadap Program MBG")
    st.markdown(
        "Analisis emosi menggunakan **NRC Emotion Lexicon versi Bahasa Indonesia** "
        "(8 emosi dasar) pada seluruh 1.286 komentar bersih. Setiap komentar "
        "diberikan satu **emosi dominan final** berdasarkan skor tertimbang "
        "tertinggi dari kata-kata yang cocok dengan kamus NRC."
    )
    st.markdown("---")

    sentimen_filter = st.selectbox(
        "Filter berdasarkan sentimen prediksi",
        options=["Semua", "negatif", "positif"],
        format_func=lambda s: "Semua" if s == "Semua" else LABEL_DISPLAY[s],
    )

    st.subheader("Ringkasan Distribusi Emosi Dominan")

    if sentimen_filter == "Semua":
        nrc_display = nrc_summary_df.copy()
        total_komentar = nrc_display["Jumlah_Komentar_Sebagai_Emosi_Dominan"].sum()
        jml_tidak_ada = int((df["emosi_dominan"] == "Tidak Ada").sum())
    else:
        row = emosi_per_sentimen_df.loc[sentimen_filter]
        nrc_display = pd.DataFrame({
            "Emosi": EMOSI_COLS,
            "Jumlah_Komentar_Sebagai_Emosi_Dominan": [int(row.get(e, 0)) for e in EMOSI_COLS],
        })
        jml_tidak_ada = int(row.get("Tidak Ada", 0))

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = plot_emosi_dominan(nrc_display)
        st.pyplot(fig)
    with col2:
        total_shown = nrc_display["Jumlah_Komentar_Sebagai_Emosi_Dominan"].sum() + jml_tidak_ada
        nrc_display_pct = nrc_display.copy()
        nrc_display_pct["Persentase"] = (nrc_display_pct["Jumlah_Komentar_Sebagai_Emosi_Dominan"] / total_shown * 100).round(2)
        st.dataframe(
            nrc_display_pct.sort_values("Jumlah_Komentar_Sebagai_Emosi_Dominan", ascending=False).rename(columns={
                "Jumlah_Komentar_Sebagai_Emosi_Dominan": "Jumlah Komentar",
                "Persentase": "Persentase (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.metric("Komentar tanpa emosi dominan (Tidak Ada)", f"{jml_tidak_ada:,}")

    st.markdown("---")

    st.subheader("Distribusi Emosi Dominan per Kelas Sentimen")
    fig2 = plot_emosi_per_sentimen(emosi_per_sentimen_df)
    st.pyplot(fig2)

    with st.expander("Lihat tabel tabulasi silang lengkap"):
        st.dataframe(emosi_per_sentimen_df.rename(index=LABEL_DISPLAY), use_container_width=True)

    st.markdown("---")

    st.subheader("Insight Utama")
    top2 = nrc_summary_df.sort_values("Jumlah_Komentar_Sebagai_Emosi_Dominan", ascending=False).head(2)
    st.markdown(
        f"- Emosi **{top2.iloc[0]['Emosi']}** merupakan emosi dominan paling banyak "
        f"({int(top2.iloc[0]['Jumlah_Komentar_Sebagai_Emosi_Dominan'])} komentar, "
        f"{top2.iloc[0]['Persentase_Emosi_Dominan']:.2f}%), diikuti oleh "
        f"**{top2.iloc[1]['Emosi']}** ({int(top2.iloc[1]['Jumlah_Komentar_Sebagai_Emosi_Dominan'])} komentar, "
        f"{top2.iloc[1]['Persentase_Emosi_Dominan']:.2f}%)."
    )
    st.markdown(
        "- Pada kelompok sentimen **Negatif**, emosi **Marah** sangat dominan, "
        "mengindikasikan kekecewaan/kritik terkait isu anggaran dan implementasi program.\n"
        "- Pada kelompok sentimen **Positif**, emosi **Percaya** paling dominan, "
        "mencerminkan kepercayaan masyarakat terhadap manfaat program bagi "
        "kesehatan dan pendidikan anak."
    )


# =============================================================================
# HALAMAN 5: WORD CLOUD
# =============================================================================

elif page == "Word Cloud":
    st.title("Word Cloud per Kelas Sentimen")
    st.markdown(
        "Visualisasi kata-kata yang paling sering muncul pada komentar "
        "bersentimen **Negatif** dan **Positif**, berdasarkan teks hasil "
        "pra-pemrosesan (`clean_text`)."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        text_negatif = " ".join(df.loc[df["sentimen_pred"] == "negatif", "clean_text"].dropna().astype(str))
        fig = plot_wordcloud(text_negatif, "Word Cloud - Sentimen Negatif")
        if fig:
            st.pyplot(fig)
    with col2:
        text_positif = " ".join(df.loc[df["sentimen_pred"] == "positif", "clean_text"].dropna().astype(str))
        fig = plot_wordcloud(text_positif, "Word Cloud - Sentimen Positif")
        if fig:
            st.pyplot(fig)


# =============================================================================
# HALAMAN 6: EKSPLORASI DATA
# =============================================================================

elif page == "Eksplorasi Data":
    st.title("Eksplorasi Data Komentar")
    st.markdown(
        "Jelajahi data komentar secara interaktif: filter berdasarkan sentimen "
        "dan emosi dominan, lalu unduh hasilnya dalam format CSV."
    )
    st.markdown("---")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        sentimen_pilih = st.multiselect(
            "Sentimen Prediksi",
            options=sorted(df["sentimen_pred"].dropna().unique()),
            default=sorted(df["sentimen_pred"].dropna().unique()),
            format_func=lambda s: LABEL_DISPLAY.get(s, s),
        )
    with col_b:
        emosi_options = sorted(df["emosi_dominan"].dropna().unique())
        emosi_pilih = st.multiselect("Emosi Dominan", options=emosi_options, default=emosi_options)
    with col_c:
        keyword = st.text_input("Cari kata kunci pada teks komentar (opsional)")

    data_filter = df.copy()
    if sentimen_pilih:
        data_filter = data_filter[data_filter["sentimen_pred"].isin(sentimen_pilih)]
    if emosi_pilih:
        data_filter = data_filter[data_filter["emosi_dominan"].isin(emosi_pilih)]
    if keyword:
        data_filter = data_filter[data_filter["full_text"].astype(str).str.contains(keyword, case=False, na=False)]

    st.markdown(f"**Menampilkan {len(data_filter):,} dari {len(df):,} komentar**")

    display_cols = [
        "full_text", "clean_text", "sentimen", "sentimen_pred",
        "emosi_dominan", "skor_emosi_dominan", "bukti_emosi_dominan",
    ]

    st.dataframe(data_filter[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Unduh Data Terfilter (CSV)",
        data=data_filter.to_csv(index=False).encode("utf-8"),
        file_name="hasil_filter_mbg.csv",
        mime="text/csv",
    )


# =============================================================================
# HALAMAN 7: PRA-PEMROSESAN DATA (Demonstrasi 7 Langkah)
# =============================================================================

elif page == "Pra-pemrosesan Data":
    st.title("Pipeline Pra-pemrosesan Data Teks")
    st.markdown(
        "Halaman ini mendemonstrasikan **7 langkah pra-pemrosesan** yang digunakan "
        "pada penelitian ini, beserta contoh hasil transformasi teks pada setiap "
        "langkah (sesuai Tabel 3.8 pada skripsi). Anda juga dapat mencoba langkah "
        "ini dengan teks Anda sendiri pada bagian bawah halaman."
    )
    st.markdown("---")

    stemmer, stopword_list = get_stemmer_and_stopwords()
    LANGKAH_INFO = [
        ("1. Cleansing", "Menghapus URL, @mention, #hashtag, angka, emoji/karakter non-ASCII, dan tanda baca."),
        ("2. Case Folding", "Mengubah seluruh huruf menjadi huruf kecil (lowercase)."),
        ("3. Tokenisasi", "Memecah kalimat menjadi token kata individual."),
        ("4. Normalisasi Slang", "Mengubah kata tidak baku/slang menjadi bentuk baku menggunakan kamus gaul."),
        ("5. Stopword Removal", "Menghapus kata umum yang tidak bermakna, KECUALI kata negasi (tidak, bukan, jangan, belum, kurang, tanpa, tak) yang tetap dipertahankan."),
        ("6. Stemming", "Mengubah kata berimbuhan menjadi bentuk kata dasar menggunakan algoritma Nazief-Adriani (Sastrawi)."),
        ("7. Clean Text", "Menggabungkan token hasil stemming menjadi satu string `clean_text` yang siap dianalisis."),
    ]

    st.subheader("Penjelasan Tiap Langkah")
    for nama, deskripsi in LANGKAH_INFO:
        st.markdown(f"**{nama}** &mdash; {deskripsi}")

    st.markdown("---")

    st.subheader("Contoh Hasil Pra-pemrosesan (sesuai Tabel 3.8 Skripsi)")
    contoh_pilihan = st.selectbox(
        "Pilih contoh komentar",
        options=range(len(CONTOH_PREPROCESSING)),
        format_func=lambda i: CONTOH_PREPROCESSING[i]["label"],
    )
    teks_contoh = CONTOH_PREPROCESSING[contoh_pilihan]["teks"]

    hasil = preprocess_pipeline(teks_contoh, stemmer, stopword_list)

    st.markdown("**Input (komentar mentah):**")
    st.code(hasil["0_input"], language=None)

    step_table = pd.DataFrame([
        {"Langkah": "1. Cleansing", "Output": hasil["1_cleansing"]},
        {"Langkah": "2. Case Folding", "Output": hasil["2_case_folding"]},
        {"Langkah": "3. Tokenisasi", "Output": str(hasil["3_tokenisasi"])},
        {"Langkah": "4. Normalisasi Slang", "Output": str(hasil["4_normalisasi_slang"])},
        {"Langkah": "5. Stopword Removal", "Output": str(hasil["5_stopword_removal"])},
        {"Langkah": "6. Stemming", "Output": str(hasil["6_stemming"])},
        {"Langkah": "7. Clean Text (Hasil Akhir)", "Output": hasil["7_clean_text"]},
    ])
    st.dataframe(step_table, use_container_width=True, hide_index=True)

    negasi_dipertahankan = [t for t in hasil["5_stopword_removal"] if t in NEGATION_TERMS]
    if negasi_dipertahankan:
        st.success(
            f"**Penanganan Negasi**: kata negasi {negasi_dipertahankan} terdeteksi pada "
            "tahap Normalisasi Slang dan **tetap dipertahankan** pada tahap Stopword "
            "Removal (tidak dihapus), karena kata negasi krusial untuk menentukan "
            "polaritas sentimen pada tahap pelabelan InSet Lexicon."
        )
    else:
        st.info("Tidak ditemukan kata negasi (tidak, bukan, jangan, belum, kurang, tanpa, tak) pada komentar ini.")

    st.caption(
        f"Jumlah token sebelum stopword removal: {len(hasil['4_normalisasi_slang'])} | "
        f"setelah stopword removal: {len(hasil['5_stopword_removal'])} | "
        f"setelah stemming: {len(hasil['6_stemming'])}"
    )

    st.markdown("---")

    st.subheader("Coba dengan Teks Anda Sendiri")
    teks_custom = st.text_area(
        "Masukkan teks komentar",
        value="Menurut saya program MBG ini tidak buruk, malah sangat bermanfaat untuk anak sekolah!",
        height=100,
    )
    if st.button("Jalankan Pra-pemrosesan", key="btn_preprocess_demo"):
        hasil_custom = preprocess_pipeline(teks_custom, stemmer, stopword_list)
        step_table_custom = pd.DataFrame([
            {"Langkah": "1. Cleansing", "Output": hasil_custom["1_cleansing"]},
            {"Langkah": "2. Case Folding", "Output": hasil_custom["2_case_folding"]},
            {"Langkah": "3. Tokenisasi", "Output": str(hasil_custom["3_tokenisasi"])},
            {"Langkah": "4. Normalisasi Slang", "Output": str(hasil_custom["4_normalisasi_slang"])},
            {"Langkah": "5. Stopword Removal", "Output": str(hasil_custom["5_stopword_removal"])},
            {"Langkah": "6. Stemming", "Output": str(hasil_custom["6_stemming"])},
            {"Langkah": "7. Clean Text (Hasil Akhir)", "Output": hasil_custom["7_clean_text"]},
        ])
        st.dataframe(step_table_custom, use_container_width=True, hide_index=True)

        negasi_custom = [t for t in hasil_custom["5_stopword_removal"] if t in NEGATION_TERMS]
        if negasi_custom:
            st.success(f"Kata negasi terdeteksi dan dipertahankan: {negasi_custom}")


# =============================================================================
# HALAMAN 8: PENGUJIAN INTERAKTIF (END-TO-END)
# =============================================================================

elif page == "Pengujian Interaktif":
    st.title("Pengujian Interaktif: Analisis Sentimen & Emosi Komentar")
    st.markdown(
        "Masukkan komentar/teks pada kolom di bawah ini untuk melihat **seluruh "
        "proses analisis secara transparan**, mulai dari pra-pemrosesan teks, "
        "pelabelan sentimen dengan InSet Lexicon (termasuk penanganan negasi), "
        "hingga identifikasi emosi dominan menggunakan NRC Emotion Lexicon."
    )
    st.markdown("---")

    stemmer, stopword_list = get_stemmer_and_stopwords()
    inset_dict, inset_max_ngram = load_inset_lexicon()
    nrc_dict, nrc_max_ngram = load_nrc_lexicon()

    st.markdown("---")

    contoh_komentar = [
        "Alhamdulillah program MBG sangat membantu, anak-anak jadi semangat sekolah",
        "Program MBG ini tidak jelek, bahkan sangat membantu, bukan pencitraan semata",
        "Dana zakat dipakai buat MBG itu jelas bukan hal yang benar, korupsi berkedok bantuan",
        "36 Trilyun buat THR Guru, mending buat MBG saja biar rakyat tidak susah",
        "Saya kecewa, MBG belum merata dan masih banyak daerah yang belum kebagian",
    ]

    pilihan_input = st.radio(
        "Sumber teks", ["Tulis sendiri", "Pilih contoh komentar"], horizontal=True
    )
    if pilihan_input == "Pilih contoh komentar":
        teks_input = st.selectbox("Contoh komentar", options=contoh_komentar)
    else:
        teks_input = st.text_area(
            "Masukkan komentar/teks untuk dianalisis",
            value="Program MBG ini tidak jelek, bahkan sangat membantu anak-anak sekolah, bukan pencitraan semata.",
            height=100,
        )

    jalankan = st.button("🔍 Analisis Komentar", type="primary")

    if jalankan and teks_input.strip():
        hasil = preprocess_pipeline(teks_input, stemmer, stopword_list)
        clean_text = hasil["7_clean_text"]
        tokens_final = hasil["6_stemming"]

        # ---------------------------------------------------------------
        # BAGIAN 1: PRA-PEMROSESAN
        # ---------------------------------------------------------------
        st.markdown("## 1\ufe0f\u20e3 Tahap Pra-pemrosesan")
        step_table = pd.DataFrame([
            {"Langkah": "0. Input Asli", "Output": hasil["0_input"]},
            {"Langkah": "1. Cleansing", "Output": hasil["1_cleansing"]},
            {"Langkah": "2. Case Folding", "Output": hasil["2_case_folding"]},
            {"Langkah": "3. Tokenisasi", "Output": str(hasil["3_tokenisasi"])},
            {"Langkah": "4. Normalisasi Slang", "Output": str(hasil["4_normalisasi_slang"])},
            {"Langkah": "5. Stopword Removal", "Output": str(hasil["5_stopword_removal"])},
            {"Langkah": "6. Stemming", "Output": str(hasil["6_stemming"])},
            {"Langkah": "7. Clean Text (Hasil Akhir)", "Output": clean_text},
        ])
        st.dataframe(step_table, use_container_width=True, hide_index=True)

        negasi_dipertahankan = [t for t in hasil["5_stopword_removal"] if t in NEGATION_TERMS]
        if negasi_dipertahankan:
            st.success(
                f"**Kata negasi terdeteksi dan dipertahankan**: {negasi_dipertahankan}. "
                "Kata-kata ini tidak dihapus pada tahap Stopword Removal karena "
                "dipakai untuk mencocokkan frasa pada `kamus_negasi.txt`."
            )
        else:
            st.info("Tidak ada kata negasi yang terdeteksi pada komentar ini.")

        if not clean_text.strip():
            st.error(
                "Hasil `clean_text` kosong setelah pra-pemrosesan (semua kata "
                "merupakan stopword atau tidak bermakna). Proses dihentikan."
            )
            st.stop()

        st.markdown("---")

        # ---------------------------------------------------------------
        # BAGIAN 2: PELABELAN SENTIMEN (InSet Lexicon + Negasi)
        # ---------------------------------------------------------------
        st.markdown("## 2\ufe0f\u20e3 Pelabelan Sentimen (InSet Lexicon)")

        total_score, matches = score_by_inset_with_negation(teks_input, inset_dict, inset_max_ngram)
        label_sentimen = tentukan_label(total_score)

        col1, col2 = st.columns([1, 2])
        with col1:
            warna = "\U0001F7E5" if label_sentimen == "negatif" else "\U0001F7E9"
            st.metric("Total Skor InSet", f"{total_score:+.1f}")
            st.metric("Label Sentimen", f"{warna} {LABEL_DISPLAY[label_sentimen]}")
            st.caption("Aturan: skor total > 0 \u2192 Positif, skor total \u2264 0 \u2192 Negatif")

        with col2:
            if matches:
                match_df = pd.DataFrame(matches)
                match_df = match_df.rename(columns={
                    "term": "Term Setelah Preprocessing",
                    "posisi": "Posisi Token",
                    "jenis_match": "Jenis Match",
                    "frasa_negasi_asli": "Frasa Negasi Asli",
                    "padanan_makna": "Padanan Makna",
                    "kategori": "Kategori",
                    "skor_asli": "Skor Asli (InSet)",
                    "skor_akhir": "Skor Akhir",
                    "keterangan": "Keterangan",
                })
                st.markdown("**Detail Pencocokan Term InSet dan Kamus Negasi Manual**")
                st.dataframe(match_df, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada frasa negasi manual atau term InSet yang cocok (skor = 0, label = Negatif).")

        matches_dengan_negasi = [m for m in matches if m.get("jenis_match") == "negasi_manual"]
        if matches_dengan_negasi:
            st.markdown("### \U0001F504 Proses Penanganan Negasi Manual")
            for m in matches_dengan_negasi:
                st.markdown(
                    f"- Frasa **'{m['frasa_negasi_asli']}'** cocok dengan kamus negasi manual "
                    f"(term setelah preprocessing: **'{m['term']}'**). "
                    f"Padanan makna: **{m['padanan_makna']}**, kategori: **{m['kategori']}**, "
                    f"skor manual: **{m['skor_akhir']:+.1f}**."
                )
        else:
            st.caption("Tidak ada frasa pada `kamus_negasi.txt` yang cocok pada komentar ini.")

        st.markdown("---")

        # ---------------------------------------------------------------
        # BAGIAN 3: PREDIKSI MODEL SVM TERBAIK
        # ---------------------------------------------------------------
        st.markdown("## 3\ufe0f\u20e3 Prediksi Model SVM Terbaik")
        st.caption(
            "Bagian ini menggunakan model SVM terbaik/terpilih dari notebook."
        )

        best_model, best_tfidf = load_svm_model(selected_scenario)
        X_input = best_tfidf.transform([clean_text])
        pred_svm = best_model.predict(X_input)[0]
        decision_label = "-"
        if hasattr(best_model, "decision_function"):
            try:
                decision_value = best_model.decision_function(X_input)[0]
                decision_label = f"{decision_value:+.4f}"
            except Exception:
                decision_label = "-"

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Prediksi SVM", LABEL_DISPLAY.get(pred_svm, pred_svm))
        with col2:
            st.metric("Skenario", selected_scenario)
        with col3:
            st.info(
                f"Model yang digunakan: **{selected_scenario} - "
                f"{SCENARIO_INFO[selected_scenario]['Keterangan']}**. "
                f"Decision function: `{decision_label}`."
            )

        st.markdown("---")

        # ---------------------------------------------------------------
        # BAGIAN 4: ANALISIS EMOSI (NRC Emotion Lexicon)
        # ---------------------------------------------------------------
        st.markdown("## 4\ufe0f\u20e3 Analisis Emosi (NRC Emotion Lexicon)")

        hasil_nrc = score_nrc_per_comment(teks_input, nrc_dict, nrc_max_ngram)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Emosi Dominan", hasil_nrc["emosi_dominan"])
            st.metric("Skor Tertimbang", f"{hasil_nrc['skor_emosi_dominan']:.4f}")
            st.metric("Jumlah Emosi Terdeteksi", hasil_nrc["jumlah_emosi_terdeteksi"])
            st.caption(f"Metode pemilihan: {hasil_nrc['metode_pemilihan_emosi']}")

        with col2:
            fig = plot_emosi_bar_single(hasil_nrc["weighted_scores"], hasil_nrc["emosi_dominan"])
            st.pyplot(fig)

        st.markdown("### \U0001F4CC Bagian Teks yang Menunjukkan Emosi")
        if hasil_nrc["matches"]:
            match_nrc_df = pd.DataFrame(hasil_nrc["matches"]).rename(columns={
                "term": "Kata/Frasa yang Cocok",
                "jenis_match": "Jenis Match",
                "emosi": "Emosi Terkait (NRC)",
                "bobot_per_emosi": "Bobot per Emosi",
                "posisi": "Posisi Token",
            })
            st.dataframe(
                match_nrc_df[["Kata/Frasa yang Cocok", "Jenis Match", "Emosi Terkait (NRC)", "Bobot per Emosi", "Posisi Token"]],
                use_container_width=True, hide_index=True,
            )

            if hasil_nrc["bukti_emosi_dominan"] != "-":
                bukti_list = [t.strip() for t in hasil_nrc["bukti_emosi_dominan"].split(",")]
                highlighted_tokens = []
                for tok in tokens_final:
                    if tok in bukti_list:
                        highlighted_tokens.append(f"**:red[{tok}]**")
                    else:
                        highlighted_tokens.append(tok)
                st.markdown(
                    f"**Kalimat hasil pra-pemrosesan dengan kata pembentuk emosi "
                    f"'{hasil_nrc['emosi_dominan']}' ditandai:**"
                )
                st.markdown(" ".join(highlighted_tokens))
                st.caption(
                    f"Kata/frasa yang menjadi bukti emosi dominan '{hasil_nrc['emosi_dominan']}': "
                    f"**{hasil_nrc['bukti_emosi_dominan']}**"
                )

            st.markdown("### Rincian Skor 8 Emosi NRC")
            detail_emosi_df = pd.DataFrame({
                "Emosi": EMOSI_COLS,
                "Raw Count": [hasil_nrc["raw_counts"][e] for e in EMOSI_8],
                "Presence (0/1)": [hasil_nrc["presence"][e] for e in EMOSI_8],
                "Skor Tertimbang": [round(hasil_nrc["weighted_scores"][e], 4) for e in EMOSI_8],
            }).sort_values("Skor Tertimbang", ascending=False)
            st.dataframe(detail_emosi_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Tidak ditemukan kata pada `clean_text` yang cocok dengan NRC "
                "Emotion Lexicon, sehingga emosi dominan diberi label 'Tidak Ada'."
            )

        st.markdown("---")

        # ---------------------------------------------------------------
        # BAGIAN 5: RINGKASAN AKHIR
        # ---------------------------------------------------------------
        st.markdown("## \u2705 Ringkasan Hasil Analisis")
        ringkasan_cols = st.columns(3)
        ringkasan_cols[0].markdown(f"**Teks Asli:**\n\n{teks_input}")
        ringkasan_cols[1].markdown(f"**Clean Text:**\n\n`{clean_text}`")
        label_inset = tentukan_label(total_score)
        ringkasan_cols[2].markdown(f"**Label InSet:** {LABEL_DISPLAY.get(label_inset, label_inset)}")
        sentimen_emoji = "\U0001F7E9 Positif" if pred_svm == "positif" else "\U0001F7E5 Negatif"
        ringkasan_cols[2].markdown(f"**Sentimen (SVM):** {sentimen_emoji}")
        ringkasan_cols[2].markdown(f"**Emosi Dominan:** {hasil_nrc['emosi_dominan']}")

    elif jalankan:
        st.warning("Mohon masukkan teks komentar terlebih dahulu.")


# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard disusun sebagai implementasi sistem (subbab 4.9) dari skripsi "
    "Analisis Sentimen dan Emosi Publik terhadap Program MBG menggunakan SVM "
    "dan NRC Emotion Lexicon - Program Studi Teknik Informatika, "
    "Universitas Trunojoyo Madura."
)
