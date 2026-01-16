from miditok.data_augmentation import augment_midi_dataset
from pathlib import Path

# Create more data by shifting pitch up/down 1 octave and changing speed slightly
augment_midi_dataset(
    data_path=Path("midi_data"),
    pitch_offsets=[-12, 12, -1, 1], 
    velocity_offsets=[10, -10],
    out_path=Path("midi_data_augmented")
)
