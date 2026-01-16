import torch
from transformers import GPT2LMHeadModel
from miditok import REMI
from pathlib import Path


BASE_MODEL_PATH = "./output/maestro_base_model"
TOKENIZER_PATH = "./output/maestro_tokenizer"
OUTPUT_MIDI_PATH = "./output/generated_base.midi"

def generate_midi():
    # 2. Load Tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = REMI.from_pretrained(TOKENIZER_PATH)

    # 3. Load Base Model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running generation on: {device}")
    model.to(device)
    model.eval()

    # 4. Prepare Input
    # We start with the BOS (Beginning of Sequence) token to kickstart generation.
    # If your tokenizer doesn't have explicit BOS, we default to 0.
    bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer.vocab else 0
    
    # Create the initial tensor [Batch_Size=1, Seq_Len=1]
    input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)

    # 5. Generate
    print("Generating tokens (this might take a moment)...")
    
    generated_ids = model.generate(
        input_ids,
        max_length=512,
        do_sample=True,
        temperature=1.0,
        top_k=50,             # Keep only top 50 likely tokens
        top_p=0.9,            # Nucleus sampling
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["EOS_None"] if "EOS_None" in tokenizer.vocab else None
    )

    # 6. Decode back to MIDI
    print("Decoding tokens to MIDI...")
    # Convert GPU tensor back to a standard Python list/numpy array
    generated_seq = generated_ids[0].cpu().numpy()
    


    generated_midi = tokenizer.decode(generated_seq)
    
    # 7. Save

    generated_midi.dump_midi(OUTPUT_MIDI_PATH)
    print(f"✅ Saved generated MIDI to {OUTPUT_MIDI_PATH}")

if __name__ == "__main__":
    generate_midi()