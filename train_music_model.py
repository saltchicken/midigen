import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback

# ---------------------------------------------------------
# 1. SETUP & TOKENIZATION
# ---------------------------------------------------------
# Define paths
midi_paths = list(Path("midi_data").glob("*.mid"))

# Configure the tokenizer (REMI is great for pop/piano music)
# We use a standard configuration that captures Pitch, Velocity, Duration, etc.
config = TokenizerConfig(num_velocities=16, use_chords=True)
tokenizer = REMI(config)

# "Train" the tokenizer (optional for small data, but good practice)
# This builds a vocabulary based on the specific notes found in your files.
tokenizer.train(vocab_size=30000, files_paths=midi_paths)
tokenizer.save_params(Path("tokenizer.json")) # Save for later

# ---------------------------------------------------------
# 2. PREPARE DATASET
# ---------------------------------------------------------
# GPT models have a fixed "context window" (e.g., 512 tokens).
# We split your midis into chunks of this size so the model can digest them.
CONTEXT_LENGTH = 512

# Create a PyTorch Dataset directly from the MIDI files
# This automatically handles tokenization and chunking
dataset = DatasetMIDI(
    files_paths=midi_paths,
    tokenizer=tokenizer,
    max_seq_len=CONTEXT_LENGTH,
    bos_token_id=tokenizer.pad_token_id, # Beginning of sequence
    eos_token_id=tokenizer["BOS_None"],  # End of sequence (using BOS as separator)
)

# Data Collator handles padding so all batches are the same shape
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)

# ---------------------------------------------------------
# 3. INITIALIZE MODEL
# ---------------------------------------------------------
# We create a *tiny* GPT-2. Standard GPT-2 is too big for 5 files 
# and will memorize them instantly (overfitting).
model_config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=CONTEXT_LENGTH,
    n_ctx=CONTEXT_LENGTH,
    n_embd=512,   # Smaller embedding size
    n_layer=6,    # Fewer layers (standard is 12)
    n_head=8,     # Fewer heads
)

model = GPT2LMHeadModel(model_config)

# ---------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="music_model_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=50,       # High epochs because data is small
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
    learning_rate=1e-4,
    remove_unused_columns=False, # Required for MidiTok datasets
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
print("Training complete! Model saved to 'my_music_model'.")
