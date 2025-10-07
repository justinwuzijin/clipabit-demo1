
import streamlit as st
import whisper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

st.title("Audio Transcription Embeddings Browser")
st.write("Upload audio files to transcribe and compare their embeddings.")

uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav"], accept_multiple_files=True)

if uploaded_files:
    model = whisper.load_model("base")
    embeddings = []
    texts = []
    filenames = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        result = model.transcribe(temp_path)
        text = result["text"]
        texts.append(text)
        emb = model.encode(text)
        embeddings.append(emb)
        filenames.append(uploaded_file.name)
    embeddings = np.vstack(embeddings)

    st.subheader("Transcriptions")
    for name, text in zip(filenames, texts):
        st.markdown(f"**{name}:** {text}")

    st.subheader("Compare Embeddings")
    if len(embeddings) > 1:
        idx1 = st.selectbox("Select first audio", range(len(filenames)), format_func=lambda x: filenames[x])
        idx2 = st.selectbox("Select second audio", range(len(filenames)), format_func=lambda x: filenames[x], index=1)
        sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]
        st.write(f"Cosine similarity between {filenames[idx1]} and {filenames[idx2]}: {sim:.3f}")

# --- Audio/Video Transcription & Semantic Search App ---
import streamlit as st
import whisper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Audio/Video Transcription Search", page_icon=":mag:")
st.title("Audio/Video Transcription Semantic Search")
st.write("Drop a video or audio file below. The app will transcribe everything said, and you can search the transcript using the chatbox.")

uploaded_file = st.file_uploader("Upload video or audio file", type=["mp3", "wav", "mp4", "m4a", "webm", "ogg"])

if uploaded_file:
    temp_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.info("Transcribing... this may take a minute.")
    model = whisper.load_model("base")
    result = model.transcribe(temp_path, verbose=False)
    segments = result.get("segments", [])
    transcript = result["text"]
    st.success("Transcription complete!")
    st.subheader("Transcript")
    st.write(transcript)

    # Prepare segment embeddings for semantic search
    segment_texts = [seg["text"] for seg in segments]
    segment_starts = [seg["start"] for seg in segments]
    if segment_texts:
        embeddings = np.vstack([model.encode(t) for t in segment_texts])
    else:
        embeddings = np.array([model.encode(transcript)])
        segment_texts = [transcript]
        segment_starts = [0.0]

    st.subheader("Chatbox: Search the transcript")
    query = st.text_input("Ask: What was said about ...?")
    if query:
        query_emb = model.encode(query)
        sims = cosine_similarity([query_emb], embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_text = segment_texts[best_idx]
        best_time = segment_starts[best_idx]
        st.markdown(f"**Best match:** {best_text}")
        st.markdown(f"**Timestamp:** {best_time:.1f} seconds")
