import serial
import wave
import time

# ==== USER CONFIGURATION ====
SERIAL_PORT = 'COM6'  # Replace with your actual COM port (e.g., '/dev/ttyUSB0' for Linux/Mac)
BAUD_RATE = 115200
OUTPUT_FILENAME = 'apollo3_audio.wav'
SAMPLE_RATE = 16000
NUM_SAMPLES = 12000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1
# ============================

print(f"Opening serial port {SERIAL_PORT}...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
time.sleep(2)  # Wait for board to reset if needed

# Flush old input
ser.reset_input_buffer()
ser.reset_output_buffer()

print("Sending 'r' to trigger recording...")
ser.write(b'r')

# Wait for response
print("Waiting for data...")
raw_data = bytearray()
while len(raw_data) < NUM_SAMPLES * SAMPLE_WIDTH:
    chunk = ser.read(NUM_SAMPLES * SAMPLE_WIDTH - len(raw_data))
    if chunk:
        raw_data.extend(chunk)
        print(f"Received {len(raw_data)} of {NUM_SAMPLES * SAMPLE_WIDTH} bytes", end='\r')

print("\nRecording complete. Saving to WAV...")

# Save to WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(raw_data)

print(f"Audio saved as: {OUTPUT_FILENAME}")
