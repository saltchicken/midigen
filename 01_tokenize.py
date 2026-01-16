from miditok import REMI, TokenizerConfig
from pathlib import Path
import json


# 1. Configuration
config = TokenizerConfig(
    num_velocities=16,
    use_chords=True,
    use_programs=True,
    use_tempos=True,
    pitch_range=(21, 108),
)

tokenizer = REMI(config)

# 2. Train the Tokenizer (BPE)
maestro_path = Path("./input/maestro-v3.0.0-midi")
midi_paths = list(maestro_path.glob("**/*.midi"))

# Safety check for empty path
if not midi_paths:
    print(f"Warning: No MIDI files found in {maestro_path}. Please check the path.")

print("Training tokenizer (BPE)... this may take a few minutes.")
tokenizer.train(vocab_size=20000, files_paths=midi_paths) 
tokenizer.save_pretrained("./output/maestro_tokenizer")


# We manually tokenize and split files into chunks for the Transformer
save_dir = Path("./output/tokenized_data_chunks")
save_dir.mkdir(parents=True, exist_ok=True)

def preprocess_dataset(files, tokenizer, out_dir, max_len=1024):
    print(f"Tokenizing and splitting {len(files)} files...")
    chunk_count = 0
    
    for midi_path in files:
        try:
            # Tokenize: Returns a TokSequence (or list of them if multi-track)
            tokens_obj = tokenizer(midi_path)
            
            # Extract IDs from the object
            full_ids = []
            if isinstance(tokens_obj, list):
                # If multiple tracks, flatten them (simple approach)
                for seq in tokens_obj:
                    full_ids.extend(seq.ids)
            else:
                # Single track
                full_ids = tokens_obj.ids

            # Split into chunks of max_len
            # We ignore the last chunk if it's too small (< 10 tokens)
            for i in range(0, len(full_ids), max_len):
                chunk = full_ids[i : i + max_len]
                if len(chunk) < 10: 
                    continue

                # Save as JSON for the Dataset class to read later
                save_path = out_dir / f"chunk_{chunk_count}.json"
                with open(save_path, "w") as f:
                    json.dump({"ids": chunk}, f)
                
                chunk_count += 1
                
        except Exception as e:
            print(f"Skipping {midi_path.name} due to error: {e}")

    print(f"Done! Created {chunk_count} chunks in {out_dir}")


preprocess_dataset(midi_paths, tokenizer, save_dir, max_len=1024)