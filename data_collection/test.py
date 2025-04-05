import serial
import numpy as np
import wave
import struct
import argparse
import serial.tools.list_ports
import os
import datetime
import uuid

# Default values
DEFAULT_BAUD = 500000  # Match the baud rate with the Artemis RedBoard ATP code
DEFAULT_LABEL = "audio"

# Parse arguments
parser = argparse.ArgumentParser(description="Serial Audio Data Collection")
parser.add_argument('-p', '--port', dest='port', type=str, required=True, help="Serial port to connect to")
parser.add_argument('-b', '--baud', dest='baud', type=int, default=DEFAULT_BAUD, help="Baud rate (default = " + str(DEFAULT_BAUD) + ")")
parser.add_argument('-d', '--directory', dest='directory', type=str, default=".", help="Output directory for files (default =.)")
parser.add_argument('-l', '--label', dest='label', type=str, default=DEFAULT_LABEL, help="Label for files (default = " + DEFAULT_LABEL + ")")

# Print out available serial ports
print()
print("Available serial ports:")
available_ports = serial.tools.list_ports.comports()
for port, desc, hwid in sorted(available_ports):
    print(" {} : {} [{}]".format(port, desc, hwid))

# Parse arguments
args = parser.parse_args()
port = args.port
baud = args.baud
out_dir = args.directory
label = args.label

print(f"Connected to: {port} successfully")

# Configure serial port
ser = serial.Serial()
ser.port = port
ser.baudrate = baud

# Attempt to connect to the serial port
try:
    ser.open()
except Exception as e:
    print("ERROR:", e)
    exit()

# Make output directory
try:
    os.makedirs(out_dir)
except FileExistsError:
    pass

# Audio configuration
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit samples
SAMPLE_RATE = 32000  # Match the sample rate of the Artemis RedBoard ATP
CHUNK_SIZE = 512  # Number of samples per chunk (match the Arduino buffer size)
RECORD_DURATION = 5  # Duration of each audio file in seconds

def main():
    buffer = []

    while True:
        # Read a chunk of audio data from the serial port
        data = ser.read(CHUNK_SIZE * SAMPLE_WIDTH)
        if not data:
            continue

        # Convert the raw data to 16-bit samples
        samples = np.frombuffer(data, dtype=np.int16)

        buffer.extend(samples)  # Add samples to the buffer

        # Check if we have enough samples for a 2-second audio file
        if len(buffer) >= SAMPLE_RATE * RECORD_DURATION * CHANNELS:
            # Generate a unique filename for the audio file
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            uid = str(uuid.uuid4())[-12:]
            filename = f"{label}.{uid}.{timestamp}.wav"
            audio_path = os.path.join(out_dir, filename)

            # Save the audio data to a WAV file
            buffer_array = np.asarray(buffer, dtype=np.int16)  # Convert buffer to an array of int16
            with wave.open(audio_path, "w") as wavefile:
                wavefile.setparams((CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, len(buffer_array), "NONE", "NONE"))
                wav_data = struct.pack("<" + ("h" * len(buffer_array)), *buffer_array)
                wavefile.writeframes(wav_data)

            print(f"Audio file '{audio_path}' saved.")
            buffer = []  # Clear the buffer after saving the audio file

if __name__ == "__main__":
    main()
