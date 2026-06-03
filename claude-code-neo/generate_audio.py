#!/usr/bin/env python3
"""Generate test audio with Harvard sentences for STT evaluation."""
import os
import sys
from gtts import gTTS
import subprocess
import tempfile

# Output path
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_audio.wav")

# Harvard sentences - well-known phonetically balanced English sentences
# These are standard test sentences for ASR evaluation
reference_text = (
    "The stale smell of old beer lingers. "
    "It takes heat to bring out the odor. "
    "A cold dip restores health and zest. "
    "A salt pickle tastes fine with ham. "
    "Tacos al pastor are my favorite. "
    "A zestful food is the hot cross bun."
)

print(f"Reference text ({len(reference_text)} chars, ~{len(reference_text.split())} words):")
print(f"  {reference_text[:80]}...")
print()

# Generate speech using gTTS
print("Generating speech with gTTS (English)...")
tts = gTTS(text=reference_text, lang='en', slow=False)

# Save to temporary MP3
with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
    mp3_path = tmp.name
tts.save(mp3_path)
print(f"  MP3 saved to: {mp3_path}")

# Convert to mono 16kHz WAV using ffmpeg or sox
print("Converting to mono 16kHz WAV...")

# Try ffmpeg first
try:
    subprocess.run(
        ['ffmpeg', '-y', '-i', mp3_path, '-ac', '1', '-ar', '16000',
         '-sample_fmt', 's16', output_path],
        capture_output=True, check=True
    )
    print(f"  Converted with ffmpeg -> {output_path}")
except (subprocess.CalledProcessError, FileNotFoundError):
    # Try sox
    try:
        subprocess.run(
            ['sox', mp3_path, '-c', '1', '-r', '16000', '-b', '16', output_path],
            capture_output=True, check=True
        )
        print(f"  Converted with sox -> {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Use Python with pydub or soundfile as fallback
        print("  ffmpeg/sox not available, trying Python fallback...")
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format="wav")
            print(f"  Converted with pydub -> {output_path}")
        except ImportError:
            print("  pydub not available, trying librosa...")
            try:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(mp3_path, sr=16000, mono=True)
                sf.write(output_path, y, 16000)
                print(f"  Converted with librosa -> {output_path}")
            except ImportError:
                print("  No audio conversion library available!")
                sys.exit(1)

# Verify output
import struct
with open(output_path, 'rb') as f:
    header = f.read(44)
    channels = struct.unpack('<H', header[22:24])[0]
    sample_rate = struct.unpack('<I', header[24:28])[0]
    bits_per_sample = struct.unpack('<H', header[34:36])[0]
    file_size = os.path.getsize(output_path)
    duration = file_size / (sample_rate * channels * (bits_per_sample // 8))

print()
print(f"Output file: {output_path}")
print(f"  Channels: {channels} (expected: 1)")
print(f"  Sample rate: {sample_rate} Hz (expected: 16000)")
print(f"  Bit depth: {bits_per_sample} (expected: 16)")
print(f"  File size: {file_size} bytes")
print(f"  Duration: {duration:.2f} seconds")

# Save reference text alongside
ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.txt")
with open(ref_path, 'w') as f:
    f.write(reference_text)
print(f"Reference text saved to: {ref_path}")

# Clean up temp MP3
os.unlink(mp3_path)
print("\nDone!")