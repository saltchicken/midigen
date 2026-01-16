import os
from pathlib import Path
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from symusic import Score  # ‼️ CHANGE: Added for preprocessing

# ---------------------------------------------------------
# 1. PREPROCESSING (Chrono Trigger -> Melody)
# ---------------------------------------------------------
# ‼️ CHANGE: Define source and destination folders
composer_raw_path = Path("midi_data_chrono_trigger")
composer_processed_path = Path("midi_data_chrono_trigger_processed")
composer_processed_path.mkdir(exist_ok=True)

print(f"Preprocessing {composer_raw_path} to match Base Model format...")
raw_files = list(composer_raw_path.glob("*.mid"))

if not raw_files:
    print(f"⚠️ Error: No MIDI files found in '{composer_raw_path}'.")
    exit()

for midi_file in raw_files:
    try:
        score = Score(str(midi_file))
        # Filter for melodic tracks (no drums, has notes)
        valid_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
        
        if valid_tracks:
            # ‼️ CHANGE: Extract the busiest track (Melody)
            melody_track = max(valid_tracks, key=lambda t: len(t.notes))
            
            new_score = Score()
            new_score.ticks_per_quarter = score.ticks_per_quarter
            new_score.tracks.append(melody_track)
            
            # Save to processed folder
            new_score.dump_midi(str(composer_processed_path / midi_file.name))
    except Exception as e:
        print(f"Skipping {midi_file.name}: {e}")

# ‼️ CHANGE: Train on the PROCESSED files, not the raw ones
midi_paths = list(composer_processed_path.glob("*.mid"))
print(f"Training LoRA on {len(midi_paths)} processed melody files.")

# ---------------------------------------------------------
# 2. SETUP MODEL
# ---------------------------------------------------------
base_model_path = "my_music_model" 
print(f"Loading base model from {base_model_path}...")

# Load tokenizer from the base model (ensures settings match)
tokenizer = REMI(params=Path("tokenizer.json"))
model = GPT2LMHeadModel.from_pretrained(base_model_path)

# ---------------------------------------------------------
# 3. APPLY LORA
# ---------------------------------------------------------
print("Applying LoRA adapter...")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            
    lora_alpha=32,  
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# ---------------------------------------------------------
# 4. PREPARE DATASET
# ---------------------------------------------------------
CONTEXT_LENGTH = 512

dataset = DatasetMIDI(
    files_paths=midi_paths,
    tokenizer=tokenizer,
    max_seq_len=CONTEXT_LENGTH,
    bos_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer["BOS_None"],
)

collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)

# ---------------------------------------------------------
# 5. TRAIN ADAPTER
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="music_model_lora_chrono",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    remove_unused_columns=False,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

print("Starting LoRA training...")
trainer.train()

model.save_pretrained("my_music_model_lora_chrono")
print("LoRA adapter saved to 'my_music_model_lora_chrono'.")
