import serial
import numpy as np
import pandas as pd
import wave
import time
from datetime import datetime

PORT = 'COM6'
BAUD = 921600
BUFFER_SIZE = 256
SAMPLE_RATE = 16000
SAVE_INTERVAL = 10

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

HEADER = b'\xAA\x55'


def save_file(audio_all, data_rows):
    if len(audio_all) == 0:
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== SAVE WAV =====
    wav_filename = f"output_{timestamp_str}.wav"
    audio_np = np.array(audio_all, dtype=np.int16)

    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_np.tobytes())

    # ===== SAVE CSV =====
    csv_filename = f"data_{timestamp_str}.csv"
    df = pd.DataFrame(data_rows, columns=["time", "audio", "x", "y", "z"])
    df.to_csv(csv_filename, index=False)

    print(f"✅ Saved: {wav_filename} & {csv_filename}")


audio_all = []
data_rows = []

print("Recording... Press Ctrl+C to stop safely")

start_chunk_time = time.time()

try:
    while True:

        # ===== tìm header =====
        while True:
            b1 = ser.read(1)
            if b1 == b'\xAA':
                b2 = ser.read(1)
                if b2 == b'\x55':
                    break

        # ===== đọc audio =====
        audio_bytes = ser.read(BUFFER_SIZE * 2)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # ===== đọc IMU =====
        imu_bytes = ser.read(12)
        x, y, z = np.frombuffer(imu_bytes, dtype=np.float32)

        block_time = time.time()
        dt = 1.0 / SAMPLE_RATE

        # ===== lưu =====
        for i in range(len(audio)):
            t = block_time + i * dt
            data_rows.append([t, audio[i], x, y, z])

        audio_all.extend(audio)
        if len(audio_all) % 16000 == 0:
            print(f"{len(audio_all)/16000:.2f} seconds")

        # ===== check 10s =====
        if len(audio_all) >= SAMPLE_RATE * SAVE_INTERVAL:
            save_file(audio_all, data_rows)

            # reset buffer
            audio_all = []
            data_rows = []
            start_chunk_time = time.time()

# ===== Ctrl + C =====
except KeyboardInterrupt:
    print("\n🛑 Stopping... Saving remaining data")
    save_file(audio_all, data_rows)

finally:
    ser.close()
    print("Serial closed")
