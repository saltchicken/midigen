from miditok import REMI
from transformers import GPT2LMHeadModel
from pathlib import Path
import torch
import random
import time

# 1. Load Model
tokenizer = REMI(params=Path("tokenizer.json"))
model = GPT2LMHeadModel.from_pretrained("my_music_model")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CHUNK_SIZE = 128        # How many tokens to generate at a time (keep small for speed)
MEMORY_SIZE = 256       # How many previous tokens the model "remembers" to keep continuity
MAX_CHUNKS = 20         # How many chunks to generate (set to 1000 for "infinite")
OUTPUT_DIR = Path("infinite_music_stream")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# PROCEDURAL STATE
# ---------------------------------------------------------
# ‼️ PROCEDURAL: Initialize with the standard start token
current_context = torch.tensor([[tokenizer.pad_token_id]])
print(f"Starting Procedural Generation into '{OUTPUT_DIR}'...")

for i in range(MAX_CHUNKS):
    # ‼️ PROCEDURAL: Randomize "Mood" every chunk
    # This creates distinct sections (e.g., a "Safe" verse vs. a "Wild" bridge)
    current_temp = random.uniform(0.9, 1.3)
    current_top_p = random.uniform(0.85, 0.98)
    
    print(f"\n[Chunk {i+1}/{MAX_CHUNKS}] Mood: Temp={current_temp:.2f}, Top_P={current_top_p:.2f}")

    # Generate the next chunk
    # ‼️ KEY CHANGE: We pass 'current_context' which contains the end of the PREVIOUS chunk
    new_history = model.generate(
        input_ids=current_context,
        max_new_tokens=CHUNK_SIZE,
        do_sample=True,
        temperature=current_temp,
        top_p=current_top_p,
        top_k=40,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"]
    )
    
    # ---------------------------------------------------------
    # SLIDING WINDOW LOGIC
    # ---------------------------------------------------------
    # 1. Get ONLY the new tokens (exclude the history we passed in)
    # The model returns [old_tokens + new_tokens]. We slice to get just the new ones.
    new_tokens_only = new_history[0, current_context.shape[1]:]
    
    # 2. Update context for the NEXT loop
    # We keep the last MEMORY_SIZE tokens from the *entire* history so the song stays coherent.
    # If we didn't do this, the model would forget the key/rhythm it was playing in.
    current_context = new_history[:, -MEMORY_SIZE:]
    
    # ---------------------------------------------------------
    # SAVE CHUNK
    # ---------------------------------------------------------
    tokens_list = new_tokens_only.tolist()
    
    # Filter special tokens
    special_ids = {tokenizer.pad_token_id, tokenizer["BOS_None"]}
    if "EOS_None" in tokenizer.vocab:
        special_ids.add(tokenizer["EOS_None"])
    
    valid_ids = [t for t in tokens_list if t not in special_ids]
    
    if len(valid_ids) > 0:
        # Save this specific chunk
        generated_midi = tokenizer.decode([valid_ids])
        filename = OUTPUT_DIR / f"chunk_{i:03d}.mid"
        generated_midi.dump_midi(str(filename))
        print(f"  -> Saved {filename} ({len(valid_ids)} notes)")
    else:
        print("  -> Chunk was empty (silence).")

print("\nGeneration Complete.")
print(f"To play continuously, drag all files in '{OUTPUT_DIR}' into VLC or use a playlist.")
