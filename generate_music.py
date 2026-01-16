from miditok import REMI
from transformers import GPT2LMHeadModel
from pathlib import Path
import torch
# ‼️ CHANGE: Import PeftModel to load the LoRA adapter
from peft import PeftModel

# 1. Load the trained model and tokenizer
tokenizer = REMI(params=Path("tokenizer.json"))

print("Loading Base Model...")
# Load the generic "Master of Melody" base model
model = GPT2LMHeadModel.from_pretrained("my_music_model")

# ‼️ CHANGE: Load and attach the LoRA Adapter
# This applies the specific "Chrono Trigger" style on top of the base model.
# print("Loading LoRA Adapter (Chrono Trigger Style)...")
# Ensure this path matches the folder name in train_lora_adapter.py (last line)
# model = PeftModel.from_pretrained(base_model, "my_music_model_lora_chrono")

# ‼️ CHANGE: Merge weights
# This merges the LoRA layers into the base model efficiently. 
# It makes generation faster and standardizes the model object.
# model = model.merge_and_unload()

# 2. Generate
print("Generating music...")

# Align start token with training script
start_token = tokenizer.pad_token_id 

# Create an explicit starting context
input_ids = torch.tensor([[start_token]])

generated_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=500,         
    do_sample=True,             
    
    # Creativity Knobs
    top_k=80,                   
    top_p=0.95,                 
    temperature=0.5,           
    
    # Loop Prevention
    repetition_penalty=2.0,     
    
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer["BOS_None"] 
)

# 3. Save to MIDI
for i, tokens in enumerate(generated_ids):
    tokens_list = tokens.tolist()
    
    print(f"\n--- Song {i} Debug Info ---")
    print(f"Total tokens generated: {len(tokens_list)}")

    special_ids = {
        tokenizer.pad_token_id, 
        tokenizer["BOS_None"], 
    }
    if "EOS_None" in tokenizer.vocab:
        special_ids.add(tokenizer["EOS_None"])

    # Remove any token that is a special token
    valid_ids = [t for t in tokens_list if t not in special_ids]
    
    print(f"Valid musical tokens (after filtering): {len(valid_ids)}")

    if len(valid_ids) == 0:
        print("⚠️ Warning: Model generated only special tokens. No music to save.")
        continue

    # Decode the sequence
    generated_midi = tokenizer.decode([valid_ids])
    
    output_filename = f"generated_song_{i}.mid"
    generated_midi.dump_midi(output_filename)
    print(f"Saved {output_filename}")
