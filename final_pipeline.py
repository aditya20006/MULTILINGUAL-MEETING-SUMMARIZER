# -*- coding: utf-8 -*-
# Multilingual Meeting Summarizer - Final Pipeline
# -------------------------------------------------------------------------------
# Features:
# - Whisper (large-v3) STT for multilingual transcription
# - fastText language detection
# - NLLB-200 for translation
# - T5 for summarization
# - BERT-NER for action item extraction
# - ROUGE & WER for evaluation
# - Gradio UI
# -------------------------------------------------------------------------------

import os
import torch
import whisper
import fasttext
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr
import evaluate

# ------------------------------
# SECTION 0 — Setup & Model Loading
# ------------------------------
print("Setting up models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Whisper
print("Loading Whisper model (large-v3)...")
model_whisper = whisper.load_model("large-v3", device=device)

# 2. Load FastText
# Ensure lid.176.bin is available. If not, download it.
if not os.path.exists("lid.176.bin"):
    print("Downloading fastText language identification model...")
    os.system("wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
lid_model = fasttext.load_model("lid.176.bin")

# 3. Load NLLB (Translation)
print("Loading NLLB model...")
model_nllb_name = "facebook/nllb-200-distilled-600M"
tokenizer_nllb = AutoTokenizer.from_pretrained(model_nllb_name)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(model_nllb_name).to(device)

# 4. Load Summarizer (T5)
print("Loading Summarization model...")
model_sum_name = "t5-small"
tokenizer_sum = T5Tokenizer.from_pretrained(model_sum_name)
model_sum = T5ForConditionalGeneration.from_pretrained(model_sum_name).to(device)

# 5. Load NER
print("Loading NER model...")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple", device=0 if device=="cuda" else -1)

# 6. Load Metrics
print("Loading Evaluation metrics...")
rouge = evaluate.load('rouge')
wer = evaluate.load('wer')

# Helper: Language Detection
def detect_language(text):
    if not text.strip():
        return "und"
    pred = lid_model.predict(text)
    lang = pred[0][0].replace('__label__', '')
    return lang

# Helper: Translation
def translate_text(text, src_lang):
    # Simple mapping from fastText/Whisper codes to NLLB codes
    # This is a subset; expand as needed
    lang_map = {
        'hi': 'hin_Deva',
        'te': 'tel_Telu',
        'en': 'eng_Latn',
        'fr': 'fra_Latn',
        'de': 'deu_Latn',
        'es': 'spa_Latn'
    }
    src_code = lang_map.get(src_lang, 'eng_Latn')
    
    if src_code == 'eng_Latn':
        return text
    
    translator = pipeline("translation", model=model_nllb, tokenizer=tokenizer_nllb, src_lang=src_code, tgt_lang="eng_Latn", max_length=512, device=0 if device=="cuda" else -1)
    result = translator(text)
    return result[0]['translation_text']

# Helper: Summarization
def summarize_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer_sum(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model_sum.generate(inputs["input_ids"], max_length=150, min_length=30, num_beams=4, early_stopping=True)
    return tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)

# Helper: Action Items
def extract_action_items(text):
    action_items = []
    entities = ner_pipeline(text)
    for entity in entities:
        if entity['entity_group'] == 'PER' or entity['entity_group'] == 'ORG':
            action_items.append(f"{entity['word']} mentioned")
    return list(set(action_items)) # Dedup

# ------------------------------
# SECTION 1 — Main Pipeline Function
# ------------------------------
def process_meeting_audio(audio_path):
    print(f"\nProcessing {audio_path}...")
    
    # 1. Transcribe
    result = model_whisper.transcribe(audio_path)
    transcription = result["text"]
    print(f"Transcription: {transcription[:100]}...")
    
    # 2. Detect Language
    lang = detect_language(transcription)
    print(f"Detected Language: {lang}")
    
    # 3. Translate
    if lang != 'en' and lang != 'und':
        print("Translating to English...")
        english_text = translate_text(transcription, lang)
        print(f"Translation: {english_text[:100]}...")
    else:
        english_text = transcription
        
    # 4. Summarize
    print("Summarizing...")
    summary = summarize_text(english_text)
    print(f"Summary: {summary}")
    
    # 5. Action Items
    print("Extracting Action Items...")
    actions = extract_action_items(english_text) # Extract from full text or summary? Usually full text is better but summary is faster. Using full text here.
    
    return transcription, lang, english_text, summary, actions

# ------------------------------
# SECTION 2 — Gradio UI
# ------------------------------
def gradio_interface(audio_file):
    transcription, lang, translation, summary, actions = process_meeting_audio(audio_file)
    return transcription, lang, translation, summary, "\n".join(actions)

if __name__ == "__main__":
    print("\nLaunching Gradio Interface...")
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Audio(type="filepath"),
        outputs=[
            gr.Textbox(label="Original Transcription"),
            gr.Textbox(label="Detected Language"),
            gr.Textbox(label="English Translation"),
            gr.Textbox(label="Meeting Summary"),
            gr.Textbox(label="Action Items / Entities")
        ],
        title="Multilingual Meeting Summarizer",
        description="Upload a meeting recording (wav/mp3) to get a summary and action items."
    )
    iface.launch(share=True)
