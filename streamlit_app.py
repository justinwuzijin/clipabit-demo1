
import streamlit as st
import whisper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

st.title("Audio Transcription Embeddings Browser")
st.write("Upload audio files to transcribe and compare their embeddings.")

uploaded_files = st.file_uploader("Upload audio or video files", type=["mp3", "wav", "mp4"], accept_multiple_files=True)

if uploaded_files:
    model = whisper.load_model("base")
    embeddings = []
    texts = []
    filenames = []

    for uploaded_file in uploaded_files:
        temp_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.info("Transcribing... this may take a minute.")
        t0 = time.time()
        model = whisper.load_model("base")
        result = model.transcribe(temp_path, verbose=False)
        t1 = time.time()
        transcript = result["text"]
        segments = result.get("segments", [])
        st.success("Transcription complete!")
        st.subheader("Transcript")
        st.write(transcript)

        # Prepare segment embeddings for semantic search
        segment_texts = [seg["text"] for seg in segments]
        segment_starts = [seg["start"] for seg in segments]
        embed_start = time.time()
        v_texts = segment_texts if segment_texts else [transcript]
        v_text_embs = np.vstack([model.encode(t) for t in v_texts])
        v_audio_embs = v_text_embs  # Placeholder: use same as text
        v_visual_embs = v_text_embs # Placeholder: use same as text
        embed_end = time.time()

        # --- Retrieval Strategy Selection ---
        st.subheader("Choose Retrieval Strategy")
        algo = st.radio(
            "Select search algorithm:",
            [
                "Average Fusion (baseline)",
                "Tiny LLM Select (hard selection)",
                "Tiny LLM Weighted (soft selection)"
            ]
        )


        # --- Chatbox Component ---
        def chatbox_component():
            st.markdown("---")
            st.subheader(":speech_balloon: Chatbox: Semantic Search")
            st.write("Type your search query below. The app will use semantic search to find the most relevant part of the transcript.")
            query = st.text_input("Type your search query and press Enter:")
            if algo and query:
                t_search_start = time.time()
                # Simulate query embeddings for all modalities
                q_text = model.encode(query)
                q_audio = q_text  # Placeholder
                q_visual = q_text # Placeholder

                scores = None
                best_idx = None
                if algo == "Average Fusion (baseline)":
                    s_v = cosine_similarity([q_visual], v_visual_embs)[0]
                    s_a = cosine_similarity([q_audio], v_audio_embs)[0]
                    s_t = cosine_similarity([q_text], v_text_embs)[0]
                    scores = (s_v + s_a + s_t) / 3
                    best_idx = int(np.argmax(scores))
                elif algo == "Tiny LLM Select (hard selection)":
                    choice = 'text' # Replace with LLM/classifier output
                    if choice == 'visual':
                        scores = cosine_similarity([q_visual], v_visual_embs)[0]
                    elif choice == 'audio':
                        scores = cosine_similarity([q_audio], v_audio_embs)[0]
                    else:
                        scores = cosine_similarity([q_text], v_text_embs)[0]
                    best_idx = int(np.argmax(scores))
                elif algo == "Tiny LLM Weighted (soft selection)":
                    weights = {'visual':0.5, 'audio':0.3, 'text':0.2} # Replace with LLM output
                    s_v = cosine_similarity([q_visual], v_visual_embs)[0]
                    s_a = cosine_similarity([q_audio], v_audio_embs)[0]
                    s_t = cosine_similarity([q_text], v_text_embs)[0]
                    scores = weights['visual']*s_v + weights['audio']*s_a + weights['text']*s_t
                    best_idx = int(np.argmax(scores))

                st.markdown("---")
                st.markdown(f"**Best match:** {v_texts[best_idx]}")
                st.markdown(f"**Timestamp:** {segment_starts[best_idx] if segment_starts else 0.0:.1f} seconds")

                # --- Metrics ---
                t_search_end = time.time()
                metrics = {
                    "Quality (subjective)": "Good", # Could be user-rated
                    "Latency (s)": f"{(t1-t0)+(t_search_end-t_search_start):.2f}",
                    "Embedding time (s)": f"{embed_end-embed_start:.2f}",
                    "Storage (MB)": f"{v_text_embs.nbytes/1e6:.2f}",
                    "Computational Intensity": "Low (demo)"
                }

                st.sidebar.header("Metrics")
                for k, v in metrics.items():
                    st.sidebar.write(f"**{k}:** {v}")
        chatbox_component()
