import serial
import wave
import time
from tqdm import tqdm

# ----- CONFIG -----
PORT = 'COM6'  # Use your port name (e.g., COM3 on Windows)
BAUD = 500000
DURATION_SECONDS = 5
SAMPLE_RATE = 15625  # ~15.625kHz as in Arduino comment
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1
WAV_FILENAME = 'artemis_recording.wav'

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Recording...")

    bytes_per_second = SAMPLE_RATE * SAMPLE_WIDTH
    total_bytes = DURATION_SECONDS * bytes_per_second
    data = bytearray()

    start = time.time()
    with tqdm(total=total_bytes, unit='B', unit_scale=True) as pbar:
        while len(data) < total_bytes:
            chunk = ser.read(1024)
            data.extend(chunk)
            pbar.update(len(chunk))

    ser.close()
    print(f"Captured {len(data)} bytes in {time.time() - start:.2f} seconds.")

    # Save to WAV
    with wave.open(WAV_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data)

    print(f"WAV saved as {WAV_FILENAME}")

if __name__ == '__main__':
    main()
