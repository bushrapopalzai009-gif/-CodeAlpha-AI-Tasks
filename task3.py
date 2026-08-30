"""
================================================================================
AI MUSIC GENERATOR - FULL PRODUCTION CODE
================================================================================
A complete deep learning pipeline for AI music generation using PyTorch LSTM.
No external dependencies needed except PyTorch.

HOW TO USE:
1. pip install torch
2. python ai_music_generator.py
3. Open the generated .mid file in any music player / DAW

EDIT THE "CONFIG" SECTION BELOW TO TUNE YOUR OUTPUT.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import struct
import os

# ==============================================================================
# ========================== CONFIG - EDIT THESE LINES =========================
# ==============================================================================

SEED = 42               # Random seed for reproducibility
SEQ_LEN = 50            # How many previous notes the model sees (memory window)
                        # Higher = longer musical memory, but slower training

# --- TRAINING ---
EPOCHS = 30             # Training rounds. 10-20 = quick demo, 50-100 = quality
BATCH_SIZE = 128        # Notes processed together. 64-256 is typical
LR = 0.003              # Learning rate. Lower = slower but more precise
HIDDEN_SIZE = 256       # LSTM brain size. 128 = small/fast, 512 = powerful
NUM_LAYERS = 3          # LSTM depth. 2 = simple, 3-4 = complex patterns
DROPOUT = 0.25          # Prevents overfitting. 0.2-0.4 is good

# --- GENERATION ---
TEMPERATURE = 0.75      # CREATIVITY DIAL:
                        #   0.3 = Very repetitive, robotic, safe
                        #   0.6 = Classical, structured, Bach-like
                        #   0.8 = Balanced, musical (RECOMMENDED)
                        #   1.2 = Jazz/improvisation, surprising
                        #   1.8+ = Chaotic, avant-garde, atonal

GEN_LEN = 400           # How many notes to generate. 200 = short, 800 = long piece

# --- MIDI OUTPUT ---
TEMPO_BPM = 100         # Playback speed. 60 = slow, 120 = standard, 180 = fast
NOTE_DURATION = 0.4     # Seconds per note. 0.3 = staccato, 0.5 = legato, 0.8 = ambient
OUTPUT_FILE = "ai_music.mid"

# --- DATA ---
NUM_TRAINING_PIECES = 200   # How many synthetic pieces to train on. 100 = fast, 500 = rich

# ==============================================================================
# ======================== DO NOT EDIT BELOW THIS LINE =========================
# ==============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("     AI MUSIC GENERATOR - Deep Learning LSTM Pipeline")
print("=" * 60)

# ==============================================================================
# STEP 1: SYNTHETIC CLASSICAL DATA GENERATION
# ==============================================================================

class SyntheticMusicDataset:
    """Generate realistic classical piano training data."""
    
    def __init__(self):
        # Extended scale with more octaves for richer music
        self.scale = [
            'C3','D3','E3','F3','G3','A3','B3',
            'C4','D4','E4','F4','G4','A4','B4',
            'C5','D5','E5','F5','G5','A5','B5','C6'
        ]
        
        # Rich chord vocabulary (major, minor, inversions)
        self.chords = {
            'C':   ['C4','E4','G4'],      'Cm':  ['C4','Eb4','G4'],
            'D':   ['D4','F#4','A4'],     'Dm':  ['D4','F4','A4'],
            'E':   ['E4','G#4','B4'],     'Em':  ['E4','G4','B4'],
            'F':   ['F4','A4','C5'],      'Fm':  ['F4','Ab4','C5'],
            'G':   ['G4','B4','D5'],      'Gm':  ['G4','Bb4','D5'],
            'Am':  ['A4','C5','E5'],      'A':   ['A4','C#5','E5'],
            'Bdim':['B4','D5','F5'],      'B':   ['B4','D#5','F#5'],
        }
        
        # Famous chord progressions
        self.progressions = [
            ['C','G','Am','F'],           # I-V-vi-IV (pop/classical)
            ['C','Am','F','G'],           # I-vi-IV-V (50s)
            ['C','F','G','C'],            # I-IV-V-I (authentic cadence)
            ['Am','F','C','G'],           # vi-IV-I-V
            ['C','Em','F','G'],           # I-iii-IV-V
            ['F','G','Em','Am'],          # IV-V-iii-vi
            ['Dm','G','C','Am'],          # ii-V-I-vi (jazz)
            ['C','G','Am','Em','F','C','F','G'],  # extended pop
            ['Am','Dm','E','Am'],         # minor progression
            ['C','F','Dm','G'],          # I-IV-ii-V
        ]
        
    def generate_arpeggio(self, chord_name, length=8):
        """Broken chord pattern (like harp or piano left hand)."""
        notes = self.chords[chord_name]
        pattern = []
        for i in range(length):
            pattern.append(notes[i % len(notes)])
        return pattern
    
    def generate_scale_run(self, start_idx, direction='up', length=8):
        """Scalar passage (like a violin run)."""
        pattern = []
        idx = start_idx
        for i in range(length):
            if 0 <= idx < len(self.scale):
                pattern.append(self.scale[idx])
            if direction == 'up':
                idx += 1
                if idx >= len(self.scale):
                    direction = 'down'
                    idx -= 2
            else:
                idx -= 1
                if idx < 0:
                    direction = 'up'
                    idx += 2
        return pattern
    
    def generate_melody(self, base_notes, length=16):
        """Melodic phrase with passing tones and ornamentation."""
        melody = []
        scale_indices = [self.scale.index(n) for n in base_notes if n in self.scale]
        
        for i in range(length):
            if random.random() < 0.25 and melody:
                # Add passing tone (stepwise motion)
                prev_idx = self.scale.index(melody[-1]) if melody[-1] in self.scale else 7
                direction = random.choice([-1, 1])
                new_idx = prev_idx + direction
                if 0 <= new_idx < len(self.scale):
                    melody.append(self.scale[new_idx])
                else:
                    melody.append(random.choice(base_notes))
            elif random.random() < 0.15:
                # Repeat previous note (sustain)
                melody.append(melody[-1] if melody else random.choice(base_notes))
            else:
                # Choose from chord tones
                melody.append(random.choice(base_notes))
        return melody
    
    def generate_alberti(self, chord_name, length=8):
        """Alberti bass pattern (classical piano accompaniment)."""
        notes = self.chords[chord_name]
        if len(notes) >= 3:
            pattern = [notes[0], notes[2], notes[1], notes[2]]
            return [pattern[i % 4] for i in range(length)]
        return self.generate_arpeggio(chord_name, length)
    
    def create_corpus(self, num_pieces=200):
        """Generate full training corpus."""
        all_notes = []
        
        for piece in range(num_pieces):
            progression = random.choice(self.progressions)
            piece_length = random.randint(80, 300)
            piece_notes = []
            chord_idx = 0
            
            while len(piece_notes) < piece_length:
                current_chord = progression[chord_idx % len(progression)]
                
                # Choose musical pattern type
                pattern_type = random.choices(
                    ['arpeggio', 'scale_run', 'melody', 'sustain', 'alberti'],
                    weights=[25, 15, 30, 10, 20]
                )[0]
                
                if pattern_type == 'arpeggio':
                    seg = self.generate_arpeggio(current_chord, length=random.randint(4, 12))
                elif pattern_type == 'scale_run':
                    start = random.randint(2, 12)
                    seg = self.generate_scale_run(start, length=random.randint(6, 16))
                elif pattern_type == 'melody':
                    seg = self.generate_melody(self.chords[current_chord], length=random.randint(8, 24))
                elif pattern_type == 'sustain':
                    seg = [random.choice(self.chords[current_chord])] * random.randint(2, 6)
                else:  # alberti
                    seg = self.generate_alberti(current_chord, length=random.randint(8, 16))
                
                piece_notes.extend(seg)
                chord_idx += 1
            
            all_notes.extend(piece_notes[:piece_length])
        
        return all_notes

print("\n[1/6] Generating synthetic classical corpus...")
dataset = SyntheticMusicDataset()
corpus = dataset.create_corpus(num_pieces=NUM_TRAINING_PIECES)
print(f"    Total notes: {len(corpus)}")
print(f"    Unique notes: {len(set(corpus))}")
print(f"    Sample: {corpus[:15]}")

# ==============================================================================
# STEP 2: PREPROCESSING
# ==============================================================================

print("\n[2/6] Preprocessing data...")

unique_notes = sorted(set(corpus))
note_to_int = {note: i for i, note in enumerate(unique_notes)}
int_to_note = {i: note for i, note in enumerate(unique_notes)}
vocab_size = len(unique_notes)

inputs, targets = [], []
for i in range(len(corpus) - SEQ_LEN):
    seq_in = corpus[i:i + SEQ_LEN]
    seq_out = corpus[i + SEQ_LEN]
    inputs.append([note_to_int[n] for n in seq_in])
    targets.append(note_to_int[seq_out])

X = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1) / float(vocab_size)
y = torch.tensor(targets, dtype=torch.long)

print(f"    Vocabulary size: {vocab_size}")
print(f"    Training sequences: {len(X)}")
print(f"    Input shape: {X.shape}")

# ==============================================================================
# STEP 3: BUILD LSTM MODEL
# ==============================================================================

print("\n[3/6] Building LSTM model...")

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, vocab_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.bn1(out)
        out = torch.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.fc3(out)
        return out

model = MusicLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
param_count = sum(p.numel() for p in model.parameters())
print(f"    Parameters: {param_count:,}")

# ==============================================================================
# STEP 4: TRAINING
# ==============================================================================

print("\n[4/6] Training model...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

dataset_tensor = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset_tensor, batch_size=BATCH_SIZE, shuffle=True)

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_music_model.pt')
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"    Epoch [{epoch+1:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

print(f"\n    Training complete! Best loss: {best_loss:.4f}")

# Load best model
model.load_state_dict(torch.load('best_music_model.pt'))
model.eval()

# ==============================================================================
# STEP 5: GENERATE MUSIC
# ==============================================================================

print("\n[5/6] Generating music...")
print(f"    Temperature: {TEMPERATURE} | Length: {GEN_LEN} notes")

seed_sequence = corpus[:SEQ_LEN]
pattern = [note_to_int[n] for n in seed_sequence]
generated_notes = []

with torch.no_grad():
    for i in range(GEN_LEN):
        # Prepare input
        seq = torch.tensor(pattern[-SEQ_LEN:], dtype=torch.float32)
        seq = seq.unsqueeze(0).unsqueeze(-1) / float(vocab_size)
        
        # Predict
        output = model(seq)
        output = output.squeeze() / TEMPERATURE
        
        # Apply temperature scaling
        probs = torch.softmax(output, dim=0)
        
        # Sample next note
        next_idx = torch.multinomial(probs, 1).item()
        generated_notes.append(int_to_note[next_idx])
        pattern.append(next_idx)
        
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{GEN_LEN} notes...")

print(f"\n    Generated {len(generated_notes)} notes.")
print(f"    Preview: {generated_notes[:20]}")

# ==============================================================================
# STEP 6: WRITE MIDI FILE
# ==============================================================================

print("\n[6/6] Writing MIDI file...")

NOTE_MAP = {
    'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
    'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11
}

def note_to_midi(note_str):
    """Convert note string to MIDI note number."""
    if note_str[1] in '#b':
        name, octave = note_str[:2], int(note_str[2:])
    else:
        name, octave = note_str[0], int(note_str[1:])
    return 12 * (octave + 1) + NOTE_MAP[name]

def encode_varlen(value):
    """Encode integer as MIDI variable-length quantity."""
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        result.insert(0, byte)
        if not value:
            break
    return bytes(result)

def write_midi(notes, filename, tempo_bpm, note_dur):
    """Write a standard Format 1 MIDI file."""
    ticks_per_quarter = 480
    tempo = int(60000000 / tempo_bpm)
    
    events = []
    
    # Track name
    events.append((0, 'meta', 0x03, b'AI Generated Music'))
    # Set tempo
    events.append((0, 'meta', 0x51, struct.pack('>I', tempo)[1:]))
    # Time signature 4/4
    events.append((0, 'meta', 0x58, b'\x04\x02\x18\x08'))
    # Key signature (C major = 0)
    events.append((0, 'meta', 0x59, b'\x00\x00'))
    
    # Note events
    delta_time = int(ticks_per_quarter * note_dur)
    
    for note_str in notes:
        try:
            pitch = note_to_midi(note_str)
            # Note on
            events.append((0, 'on', pitch, 80))
            # Note off
            events.append((delta_time, 'off', pitch, 0))
        except (KeyError, IndexError, ValueError):
            continue
    
    # End of track
    events.append((0, 'meta', 0x2F, b''))
    
    # Build track data
    track_data = bytearray()
    
    for delta, event_type, *args in events:
        track_data.extend(encode_varlen(delta))
        
        if event_type == 'meta':
            track_data.append(0xFF)
            track_data.append(args[0])
            track_data.extend(encode_varlen(len(args[1])))
            track_data.extend(args[1])
        elif event_type == 'on':
            track_data.append(0x90)  # Note on, channel 0
            track_data.append(args[0])
            track_data.append(args[1])
        elif event_type == 'off':
            track_data.append(0x80)  # Note off, channel 0
            track_data.append(args[0])
            track_data.append(args[1])
    
    # Write file
    with open(filename, 'wb') as f:
        # Header chunk
        f.write(b'MThd')
        f.write(struct.pack('>I', 6))
        f.write(struct.pack('>H', 1))  # Format 1
        f.write(struct.pack('>H', 1))  # 1 track
        f.write(struct.pack('>H', ticks_per_quarter))
        
        # Track chunk
        f.write(b'MTrk')
        f.write(struct.pack('>I', len(track_data)))
        f.write(track_data)
    
    return os.path.abspath(filename)

output_path = write_midi(generated_notes, OUTPUT_FILE, TEMPO_BPM, NOTE_DURATION)

print(f"\n{'='*60}")
print(f"  SUCCESS! MIDI file created:")
print(f"  {output_path}")
print(f"{'='*60}")
print(f"\n  File size: {os.path.getsize(output_path)} bytes")
print(f"  Notes: {len(generated_notes)}")
print(f"  Tempo: {TEMPO_BPM} BPM")
print(f"\n  Open this file in any music player, DAW, or browser.")
print(f"{'='*60}")
model = MusicLSTM(vocab_size=9, hidden=32)
model.load_state_dict(torch.load('best_music_model.pt'))
model.eval()  # Ready to generate music!