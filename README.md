# 🎤 Real-Time Speech-to-Text ASR Web App

This is a Streamlit-based web application for **Real-Time Automatic Speech Recognition (ASR)**. The app allows users to upload `.wav` or `.flac` audio files and receive transcriptions using a trained deep learning model.

---

## 📌 Features

- ✅ Upload audio files (.wav or .flac)
- ✅ Extracts MFCC features from speech
- ✅ Predicts phoneme sequences using a TensorFlow-trained ASR model
- ✅ Decodes output using a greedy CTC decoder
- ✅ Displays waveform of the audio signal
- ✅ Real-time inference and transcription

---

## 🧠 Model Details

- **Model Type**: CTC-based RNN/GRU/LSTM model
- **Input**: MFCC features with shape `(time_steps, 13)`
- **Output**: Logits over vocabulary tokens
- **Framework**: TensorFlow 2.x
- **Saved Format**: TensorFlow SavedModel (loaded using `TFSMLayer`)

---

## 🗂 Project Structure

```
RealTime-ASR-Project/
├── app/
│   └── streamlit_app.py        # Main Streamlit app
├── models/
│   └── predict_tf/             # SavedModel directory
├── audio/                      # (Optional) Test audio samples
└── README.md                   # This file
```

---

## 🚀 How to Run

1. **Install Requirements**:
   ```bash
   pip install streamlit librosa tensorflow matplotlib
   ```

2. **Start the App**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Use the Interface**:
   - Upload a `.wav` or `.flac` file
   - See the waveform and transcription in real time

---

## 🧪 Example

| Input Audio        | Output Transcription |
|--------------------|----------------------|
| `hello_world.wav`  | `hello world`        |

---

## ⚙️ Configuration

- **Model Path**: Change the `MODEL_SAVED_DIR` variable in `streamlit_app.py` if needed
- **Max Time Steps**: Modify `MAX_TIME_STEPS` if your model expects a different input shape

---

## 📖 Glossary

- **ASR (Automatic Speech Recognition)**: Converts speech audio to text using deep learning.
- **CTC (Connectionist Temporal Classification)**: Loss function and decoding method for unaligned sequence data.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Widely used feature set for speech analysis.

---


## 📝 License

This project is licensed under the MIT License. Feel free to modify and use it as needed.
