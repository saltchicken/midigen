from miditok import REMI
from transformers import GPT2LMHeadModel
from pathlib import Path
import torch

# 1. Load the trained model and tokenizer
tokenizer = REMI(params=Path("tokenizer.json"))
model = GPT2LMHeadModel.from_pretrained("my_music_model")

# 2. Generate
print("Generating music...")

# ‼️ CHANGE: Align start token with training script
# In train_music_model.py, bos_token_id was set to tokenizer.pad_token_id.
# We must use the same token here to trigger generation correctly.
start_token = tokenizer.pad_token_id 

# ‼️ CHANGE: Create an explicit starting context
# Instead of input_ids=None, we give it the start token manually.
input_ids = torch.tensor([[start_token]])

generated_ids = model.generate(
    input_ids=input_ids,        # Start with the training's BOS token
    max_new_tokens=500,         
    do_sample=True,             
    top_k=50,
    top_p=0.9,                  # ‼️ CHANGE: Added Nucleus sampling to cut off low-probability repetitions
    temperature=1.0,            # ‼️ CHANGE: Increased temperature to 1.0 to encourage diversity
    repetition_penalty=5.0,     # ‼️ CHANGE: Greatly increased penalty to force the model to pick NEW notes
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer["BOS_None"] # ‼️ CHANGE: Use the token trained as EOS (from training script)
)

# 3. Save to MIDI
for i, tokens in enumerate(generated_ids):
    tokens_list = tokens.tolist()
    
    print(f"\n--- Song {i} Debug Info ---")
    print(f"Total tokens generated: {len(tokens_list)}")
    print(f"First 20 tokens: {tokens_list[:20]}") 

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
        print("   -> If this persists, the model might need more training epochs (e.g., 100+).")
        continue

    # Decode the sequence
    generated_midi = tokenizer.decode([valid_ids])
    
    output_filename = f"generated_song_{i}.mid"
    generated_midi.dump_midi(output_filename)
    print(f"Saved {output_filename}")
