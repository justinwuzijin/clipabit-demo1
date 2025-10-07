## Search Algorithm Selection

Users can choose which search algorithm to use for semantic retrieval:
- **Average Fusion**: Combines vectors or similarity scores for retrieval.
- **Tiny LLM Select**: Uses a small language model to select the primary modality for search.
- **Tiny LLM Weighted**: Assigns weights to each modality using a small LLM and fuses results accordingly.

## Metrics Displayed in the UI Sidebar

The sidebar provides live data on the following metrics for each search and embedding operation:
- **Quality**: Subjective measure (e.g., by virtue of having eyes, user can judge relevance).
- **Latency**: Time to embed a query, search, and merge results.
- **Embedding Time**: Per-clip embedding time (batch and single).
- **Storage**: Database size per clip and total.
- **Computational Intensity**: Resource usage for each operation.
## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/clipabit-demo1/main/streamlit_app.py)
# Audio Transcription Demo (ClipABit Demo 1)

A simple Streamlit app displaying the audio transcription -> search part of ClipABit -- semantic search engine for video editors -- pipeline.


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
