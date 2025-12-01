# train_asr.py
# Script to fine-tune Whisper-small on AMI and FLEURS (Hinglish/Indic proxy)

import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

def train():
    print("1. Loading Datasets...")
    
    # A. AMI Corpus (Meetings) - 'ihm' (Individual Headset Mic) is cleaner
    print("   - Loading AMI (IHM subset)...")
    # Note: 'edinburghcstr/ami' might require manual download approval or have issues in some envs.
    # If it fails, we fall back to a safe default, but we try it first.
    try:
        ds_ami = load_dataset("edinburghcstr/ami", "ihm", split="train[:100]", trust_remote_code=True)
        # Standardize columns: AMI has 'audio', 'text'
    except Exception as e:
        print(f"   ! Failed to load AMI: {e}. Falling back to LibriSpeech as meeting proxy.")
        ds_ami = load_dataset("librispeech_asr", "clean", split="train.100[:100]")

    # B. FLEURS (Hinglish/Indic Proxy)
    print("   - Loading FLEURS (Hindi & English)...")
    try:
        ds_hi = load_dataset("google/fleurs", "hi_in", split="train[:50]", trust_remote_code=True)
        ds_en = load_dataset("google/fleurs", "en_us", split="train[:50]", trust_remote_code=True)
    except Exception as e:
        print(f"   ! Failed to load FLEURS: {e}. Using LibriSpeech only.")
        ds_hi = None
        ds_en = None

    # Standardize and Concatenate
    # We need 'audio' and 'text' columns.
    # FLEURS has 'raw_transcription' or 'transcription'. Let's check/map.
    
    datasets_to_concat = []
    
    # Process AMI
    if "text" in ds_ami.column_names:
        ds_ami = ds_ami.select_columns(["audio", "text"])
    datasets_to_concat.append(ds_ami)

    # Process FLEURS
    if ds_hi and ds_en:
        # FLEURS usually has 'transcription' or 'raw_transcription'
        # We rename to 'text'
        col_name = "transcription" if "transcription" in ds_hi.column_names else "raw_transcription"
        ds_hi = ds_hi.rename_column(col_name, "text").select_columns(["audio", "text"])
        ds_en = ds_en.rename_column(col_name, "text").select_columns(["audio", "text"])
        datasets_to_concat.append(ds_hi)
        datasets_to_concat.append(ds_en)

    print(f"   - Combining {len(datasets_to_concat)} datasets...")
    dataset = concatenate_datasets(datasets_to_concat)
    dataset = dataset.shuffle(seed=42)

    print("2. Loading Model and Processor (whisper-small)...")
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Preprocessing
    print("3. Preprocessing Data...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        # Compute log-Mel input features
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # Encode target text
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Filter out empty audio/text if any
    dataset = dataset.filter(lambda x: x["audio"] is not None and x["text"] is not None and len(x["text"]) > 0)
    
    tokenized_datasets = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    # Split for eval
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

    # Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training Arguments
    print("4. Setting up Trainer...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./fine_tuned_whisper",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=10,
        max_steps=50, # Increased slightly for multi-dataset
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("5. Starting Training (Fine-tuning)...")
    trainer.train()

    print("6. Saving Model...")
    model.save_pretrained("./fine_tuned_whisper")
    processor.save_pretrained("./fine_tuned_whisper")
    print("Done! Model saved to ./fine_tuned_whisper")

if __name__ == "__main__":
    train()
