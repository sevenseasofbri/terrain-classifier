import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct

# Load WAV file using the wave module
wav_file = wave.open('artemis_recording_01.wav', 'rb')
framerate = wav_file.getframerate()
nframes = wav_file.getnframes()
nchannels = wav_file.getnchannels()
sampwidth = wav_file.getsampwidth()

# Read all frames
frames = wav_file.readframes(nframes)
wav_file.close()

# Unpack byte data to int
fmt = {1: 'B', 2: 'h', 4: 'i'}[sampwidth]
fmt = f"<{nframes * nchannels}{fmt}"
samples = struct.unpack(fmt, frames)

# Convert to NumPy array, keep one channel if stereo
samples = np.array(samples)
if nchannels > 1:
    samples = samples[::nchannels]

# Normalize
samples = samples / np.max(np.abs(samples))

# Matplotlib setup
fig, ax = plt.subplots(figsize=(8, 3))
line, = ax.plot([], [], lw=1)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
plt.tight_layout()

# Parameters for animation
window_size = framerate  # 1 second of audio
hop = framerate // 20    # 30 fps

def update(frame):
    start = frame * hop
    end = start + window_size
    segment = samples[start:end]
    if len(segment) < window_size:
        segment = np.pad(segment, (0, window_size - len(segment)))
    x = np.linspace(0, 1, window_size)
    line.set_data(x, segment)
    return line,

total_frames = (len(samples) - window_size) // hop

ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

# Save as GIF
ani.save("waveform.gif", writer='pillow', fps=20)

# To save as MP4:
# ani.save("waveform.mp4", writer='ffmpeg', fps=30)

print("Done! Saved as waveform.gif")
