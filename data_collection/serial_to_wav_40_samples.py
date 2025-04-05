import serial
import wave
import time
import os
from tqdm import tqdm

# ----- CONFIG -----
PORT = 'COM5'  # Update with your correct port
BAUD = 500000
DURATION_SECONDS = 5
SAMPLE_RATE = 15625  # ~15.625kHz
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1
NUM_FILES = 40
OUTPUT_DIR = 'sandpaper'

def record_wav(filename, ser, duration_seconds):
    bytes_per_second = SAMPLE_RATE * SAMPLE_WIDTH
    total_bytes = duration_seconds * bytes_per_second
    data = bytearray()

    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc=filename) as pbar:
        while len(data) < total_bytes:
            chunk = ser.read(1024)
            data.extend(chunk)
            pbar.update(len(chunk))

    # Save to WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Starting 40 recordings...")

    for i in range(31, NUM_FILES + 1):
        filename = os.path.join(OUTPUT_DIR, f'artemis_recording_{i:02d}.wav')
        print(f"\nRecording file {i} of {NUM_FILES} -> {filename}")
        record_wav(filename, ser, DURATION_SECONDS)
        time.sleep(0.5)  # Optional: short pause between recordings

    ser.close()
    print("\nAll recordings complete.")

if __name__ == '__main__':
    main()
