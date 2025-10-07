# :earth_americas: GDP dashboard template

A simple Streamlit app displaying 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

## Audio Transcription Embeddings & Search Algorithms Demo

### Project Purpose

This project demonstrates how to generate audio transcription embeddings using OpenAI Whisper, and explores different search algorithms for retrieval tasks. The goal is to experiment with various fusion and selection strategies for searching among multimodal embeddings.

### Workflow

1. **Audio Transcription & Embedding**
   - Upload audio files via the Streamlit app.
   - Transcribe each audio file using Whisper.
   - Generate embeddings for each transcription.

2. **Search Algorithms Tested**
   - **Average Fusion**: Combine all modality vectors into a single index vector (or average similarity scores) and retrieve the most relevant result.
   - **Tiny LLM Select**: Use a small language model (or classifier) to select the primary modality based on the prompt, then retrieve from that modality's index only.
   - **Tiny LLM Weighted**: Use a small language model to assign weights to each modality and fuse retrieval results using those weights.

### How to Run

1. Install dependencies:
   ```bash
   pip install openai-whisper scikit-learn streamlit
   ```
2. Start the app:
   ```bash
   streamlit run streamlit_app.py
   ```

### Next Steps

- Implement and compare the search algorithms listed above.
- Experiment with different fusion strategies and LLMs for selection/weighting.

---

Feel free to contribute or suggest improvements!
