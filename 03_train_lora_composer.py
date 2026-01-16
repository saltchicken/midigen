from peft import LoraConfig, get_peft_model, TaskType
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, split_files_for_training
from pathlib import Path

# --- CONFIGURATION ---
COMPOSER_NAME = "Chopin"
RAW_COMPOSER_MIDI_DIR = Path("./input/chopin_midis")
BASE_MODEL_PATH = "./output/maestro_base_model"
TOKENIZER_PATH = "./output/maestro_tokenizer"
# ---------------------

# 1. Load Tokenizer & Base Model
tokenizer = REMI.from_pretrained(TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)


# This freezes the base model and adds trainable rank-decomposition matrices
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # Rank: Higher = more params, closer to full finetune
    lora_alpha=32,  # Alpha usually 2x Rank
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 
# You should see ~1% trainable params, meaning very fast training!

# 3. Prepare Composer Data on the fly
# (We reuse the tokenizer but tokenize the new composer files)
chunks_dir = Path(f"./output/processed_{COMPOSER_NAME}")
composer_files = list(RAW_COMPOSER_MIDI_DIR.glob("**/*.mid"))

split_files_for_training(
    files_paths=composer_files,
    tokenizer=tokenizer,
    save_dir=chunks_dir,
    max_seq_len=1024,
)

composer_dataset = DatasetMIDI(
    files_paths=list(chunks_dir.glob("**/*.json")),
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)

# 4. Train the Adapter
training_args = TrainingArguments(
    output_dir=f"./output/lora_{COMPOSER_NAME}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    num_train_epochs=10,            # More epochs needed for small datasets
    fp16=True,
    logging_steps=50,
    save_steps=500,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=composer_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# 5. Save ONLY the LoRA adapter
model.save_pretrained(f"./output/adapters/{COMPOSER_NAME}")
print(f"LoRA adapter for {COMPOSER_NAME} saved!")

# -----------------------------------------------------------------
# ### **How to Generate Music**
# To generate, you load the base model, then "attach" the specific composer adapter you want.
#
# from peft import PeftModel
# from transformers import GPT2LMHeadModel
#
# # Load Base

#
# # Load Composer Style

#
# # ... run standard generation using tokenizer.decode ...
# -----------------------------------------------------------------