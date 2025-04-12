import serial
import time
import csv
import os

# ----- CONFIG -----
PORT = 'COM6'
BAUD = 500000
SAMPLE_RATE = 15625
DURATION_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
NUM_FILES = 35
OUTPUT_DIR = 'sandpaper'

def record_csv(filename):
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Let the port settle

    print(f"\nRecording -> {filename}")
    values = []
    start_time = time.time()

    while len(values) < NUM_SAMPLES:
        line = ser.readline().decode(errors='ignore').strip()
        if line.isdigit():
            values.append(int(line))

    elapsed = time.time() - start_time
    print(f"Recorded {len(values)} samples in {elapsed:.2f} seconds")

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for val in values:
            writer.writerow([val])

    ser.close()
    print(f"Saved to {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(1, NUM_FILES + 1):
        filename = os.path.join(OUTPUT_DIR, f'artemis_recording_{i:02d}.csv')
        record_csv(filename)
        time.sleep(0.5)  # Short pause between recordings

if __name__ == '__main__':
    main()

# import serial
# import time
# import csv

# PORT = 'COM6'
# BAUD = 500000
# SAMPLE_RATE = 15625
# DURATION_SECONDS = 5
# NUM_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
# OUTPUT_FILE = 'vib-test/artemis_recording.csv'

# def record_csv():
#     ser = serial.Serial(PORT, BAUD, timeout=1)
#     time.sleep(2)  # Let the port settle

#     print("Recording...")

#     values = []
#     start_time = time.time()

#     while len(values) < NUM_SAMPLES:
#         line = ser.readline().decode(errors='ignore').strip()
#         if line.isdigit():
#             values.append(int(line))

#     elapsed = time.time() - start_time
#     print(f"Done recording {len(values)} samples in {elapsed:.2f} seconds")

#     with open(OUTPUT_FILE, 'w', newline='') as f:
#         writer = csv.writer(f)
#         for val in values:
#             writer.writerow([val])

#     ser.close()
#     print(f"Saved to {OUTPUT_FILE}")

# if __name__ == '__main__':
#     record_csv()
