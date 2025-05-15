import streamlit as st
import numpy as np
import tensorflow as tf
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display

# === Constants ===
MODEL_SAVED_DIR = "/content/drive/MyDrive/RealTime-ASR-Project/models/predict_tf"
MAX_TIME_STEPS = 485

# === Load Inference Model via TFSMLayer ===
@st.cache_resource
def get_inference_model():
    tfsml = tf.keras.layers.TFSMLayer(MODEL_SAVED_DIR, call_endpoint='serving_default')
    inp = tf.keras.Input(shape=(MAX_TIME_STEPS, 13), name='audio_features')
    out = tfsml(inp)
    return tf.keras.Model(inputs=inp, outputs=out)

# Initialize model in session_state to avoid reloads
if 'model' not in st.session_state:
    st.session_state.model = None

# === Character mapping ===
char_to_int = {
    ' ': 0, "'": 1,
    'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9,
    'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16,
    'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23,
    'w': 24, 'x': 25, 'y': 26, 'z': 27
}
char_to_int['<blank>'] = len(char_to_int)
int_to_char = {v: k for k, v in char_to_int.items()}
blank_index = char_to_int['<blank>']

# === Feature extraction ===
@st.cache_data
def extract_features(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

# === CTC Greedy Decoder ===
def ctc_greedy_decoder(logits, int_to_char, blank_index):
    # logits shape: (batch, time_steps, vocab)
    batch_size, time_steps, _ = logits.shape
    input_lengths = np.ones(batch_size) * time_steps
    decoded, _ = tf.keras.backend.ctc_decode(logits, input_length=input_lengths, greedy=True)
    seqs = decoded[0].numpy()
    texts = []
    for seq in seqs:
        text = ''.join([int_to_char.get(i, '') for i in seq if i != -1 and i != blank_index])
        texts.append(text)
    return texts

# === Streamlit UI ===
st.set_page_config(page_title="ASR Demo", layout="wide")
st.title("Real-Time Speech-to-Text ASR Web App")
st.write("Upload a WAV or FLAC file and get the transcription.")

uploaded = st.file_uploader("Choose an audio file (.wav or .flac)", type=["wav", "flac"])
if uploaded:
    # Load model if not already
    if st.session_state.model is None:
        with st.spinner('Loading ASR model...'):
            st.session_state.model = get_inference_model()
    model = st.session_state.model

    # Save upload to temp file with correct extension
    ext = uploaded.name.split('.')[-1]
    tmp_path = f"temp_audio.{ext}"
    with open(tmp_path, 'wb') as f:
        f.write(uploaded.getbuffer())

    st.audio(tmp_path)

    # Extract and pad features
    features = extract_features(tmp_path)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [features], maxlen=MAX_TIME_STEPS, dtype='float32', padding='post', value=0.0
    )

    # Predict logits
    logits_raw = model.predict(padded)
    # Handle dict outputs
    if isinstance(logits_raw, dict):
        logits = list(logits_raw.values())[0]
    else:
        logits = logits_raw
    st.write("Logits shape:", logits.shape)

    # Decode to text
    text = ctc_greedy_decoder(logits, int_to_char, blank_index)[0]
    st.subheader("Transcription:")
    st.write(text)

    # Waveform display
    y, sr = librosa.load(tmp_path, sr=16000)
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Cleanup
    os.remove(tmp_path)
