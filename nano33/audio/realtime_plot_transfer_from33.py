import serial
import numpy as np
import wave
import time
from datetime import datetime

PORT = 'COM6'
BAUD = 921600
DURATION = 5
SAMPLE_RATE = 16000

ser = serial.Serial(PORT, BAUD)


time.sleep(2)

ser.reset_input_buffer()

time.sleep(0.1)
while ser.in_waiting:
    ser.read(ser.in_waiting)

print("Recording...")


num_samples = SAMPLE_RATE * DURATION
buffer = bytearray()

while len(buffer) < num_samples * 2:
    data = ser.read(1024)
    if data:
        buffer.extend(data)

ser.close()

audio = np.frombuffer(buffer[:num_samples*2], dtype=np.int16)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"output_{timestamp}.wav"
with wave.open(filename, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())

print("Saved OK")
