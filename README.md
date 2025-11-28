# Multilingual Meeting Summarizer

## 📂 Project Files
-   **`final_pipeline.py`**: The main Python script containing the complete pipeline (Whisper -> FastText -> NLLB -> T5 -> NER) and a Gradio interface.
-   **`Report.md`**: Detailed project report including methodology, architecture, expected outputs, and performance graphs.
-   **`requirements.txt`**: List of Python dependencies.

## 🚀 How to Run (Standalone Pipeline)
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `ffmpeg` separately if not already installed.*

2.  **Run the Script**:
    ```bash
    python final_pipeline.py
    ```
    This will launch a Gradio interface where you can upload audio files.

## 🌐 Web App Version (Alternative)
If you prefer the web application version:
1.  Run `run_app.bat`.
2.  Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 📄 Report & Outputs
See **`Report.md`** for the full project report, including:
-   System Architecture (Diagrams)
-   Methodology
-   Expected Outputs (Sample Transcriptions & Summaries)
-   Performance Graphs

