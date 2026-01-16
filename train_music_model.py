import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from symusic import Score  # Import symusic for MIDI processing

# ---------------------------------------------------------
# 0. PREPROCESSING (Maestro -> Melody)
# ---------------------------------------------------------
# ‼️ CHANGE: Point to the new downloaded dataset folder
original_data_path = Path("midi_data_base")
# ‼️ CHANGE: Output folder for processed base data
melody_data_path = Path("midi_data_base_processed")
melody_data_path.mkdir(exist_ok=True)

print("Preprocessing Base Dataset (Extracting Melodies)...")
# Support both .mid and .midi extensions (Maestro uses .midi)
midi_files = list(original_data_path.glob("*.mid")) + list(original_data_path.glob("*.midi"))

if not midi_files:
    print(f"⚠️ Error: No MIDI files found in '{original_data_path}'.")
    print("   -> Did you run 'python download_dataset.py'?")
    exit()

for midi_file in midi_files:
    try:
        # Load the song
        score = Score(str(midi_file))
        
        # Filter for tracks that are NOT drums and have notes
        valid_tracks = [t for t in score.tracks if not t.is_drum and len(t.notes) > 0]
        
        if valid_tracks:
            # Pick the track with the most notes. 
            # For Maestro (Piano), this usually captures the main performance well.
            melody_track = max(valid_tracks, key=lambda t: len(t.notes))
            
            # Create a new file with ONLY that track
            new_score = Score()
            new_score.ticks_per_quarter = score.ticks_per_quarter
            new_score.tracks.append(melody_track)
            
            # Save to the new folder
            # ‼️ CHANGE: Ensure filename ends in .mid
            output_name = midi_file.stem + ".mid"
            new_score.dump_midi(str(melody_data_path / output_name))
    except Exception as e:
        print(f"Skipping {midi_file.name}: {e}")

# Update path to point to the CLEAN data
midi_paths = list(melody_data_path.glob("*.mid"))
print(f"Training on {len(midi_paths)} processed melody files.")

# ---------------------------------------------------------
# 1. SETUP & TOKENIZATION
# ---------------------------------------------------------
# Simplified configuration for Melody
# - use_chords=False: Forces model to think in single melodic lines.
# - num_velocities=4: Reduce dynamics complexity.
config = TokenizerConfig(
    num_velocities=4, 
    use_chords=False, 
    use_programs=False 
)
tokenizer = REMI(config)

# Train the tokenizer
print("Training tokenizer...")
tokenizer.train(vocab_size=30000, files_paths=midi_paths)
tokenizer.save_params(Path("tokenizer.json"))

# ---------------------------------------------------------
# 2. PREPARE DATASET
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
# 3. INITIALIZE MODEL
# ---------------------------------------------------------
model_config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=CONTEXT_LENGTH,
    n_ctx=CONTEXT_LENGTH,
    n_embd=512,
    n_layer=6,
    n_head=8,
)

model = GPT2LMHeadModel(model_config)

# ---------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="music_model_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=5,       # ‼️ CHANGE: Maestro is huge (1200+ songs). 5 epochs is plenty for a base.
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    learning_rate=1e-4,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

print("Starting training...")
trainer.train()
trainer.save_model("my_music_model")
print("Training complete! Base model saved to 'my_music_model'.")
