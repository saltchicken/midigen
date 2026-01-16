import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from pathlib import Path
from miditok import REMI

# 1. Load the tokenizer (Native MidiTok)
miditok_tokenizer = REMI.from_pretrained("./output/maestro_tokenizer")


# REMI usually has one, but if not, we default to 0 to prevent crashes.
PAD_TOKEN_ID = miditok_tokenizer.pad_token_id if miditok_tokenizer.pad_token_id is not None else 0
VOCAB_SIZE = len(miditok_tokenizer)

print(f"Loaded Tokenizer. Vocab Size: {VOCAB_SIZE}, Pad ID: {PAD_TOKEN_ID}")


# This reads the specific JSON format {"ids": [...]} we created in step 01.
# It bypasses miditok's internal Dataset classes to ensure we read our files correctly.
class MaestroMidiDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("**/*.json"))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "r") as f:
            data = json.load(f)
        # We return a dict with 'input_ids' which is what the Trainer expects
        return {"input_ids": torch.tensor(data["ids"], dtype=torch.long)}


# This replaces the Hugging Face DataCollator. It manually pads our batches.
def collate_fn(batch):
    # batch is a list of dicts: [{'input_ids': tensor([...])}, ...]
    input_ids_list = [item["input_ids"] for item in batch]
    
    # Pad sequences to the longest in the batch
    # batch_first=True returns (Batch, Seq_Len)
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_TOKEN_ID)
    
    # Create attention mask (1 for data, 0 for pad)
    attention_mask = (padded_input_ids != PAD_TOKEN_ID).long()
    
    # For GPT-2 (Causal LM), the labels are the same as input_ids.
    # The model handles shifting internally.
    labels = padded_input_ids.clone()
    # Optional: Set pad labels to -100 so they are ignored in loss calculation
    if PAD_TOKEN_ID is not None:
        labels[labels == PAD_TOKEN_ID] = -100
        
    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 4. Create the Model
model_config = GPT2Config(
    vocab_size=VOCAB_SIZE, 
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    bos_token_id=miditok_tokenizer["BOS_None"],
    eos_token_id=miditok_tokenizer["EOS_None"],
    pad_token_id=PAD_TOKEN_ID, 
)
model = GPT2LMHeadModel(model_config)

# 5. Load Dataset
dataset = MaestroMidiDataset("./output/tokenized_data_chunks")

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./output/maestro_base_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8, 
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=1e-4,
    fp16=True, # Ensure you have CUDA available, otherwise set False
    remove_unused_columns=False,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model("./output/maestro_base_model")
print("Base model training complete!")