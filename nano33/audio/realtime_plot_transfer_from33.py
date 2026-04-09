import serial
import time

# ===== CẤU HÌNH =====
PORT = 'COM6'
BAUD_RATE = 115200
FILENAME = "audio_data.csv"

LABEL = 1             # ví dụ: 1 = ngáy, 0 = bình thường
DURATION = 60         # ghi bao nhiêu giây


ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # chờ Arduino reset

print(f"Recording audio from {PORT} → '{FILENAME}' | LABEL = {LABEL}")
print("Press Ctrl+C to stop...\n")

start_time = time.time()

# ===== GHI FILE =====
with open(FILENAME, "w") as f:
    f.write("audio,label\n")  # header đơn giản

    try:
        while time.time() - start_time < DURATION:
            line = ser.readline().decode("utf-8").strip()

            if line:
                try:
                    value = int(line)  # audio sample (int16)
                    labeled_line = f"{value},{LABEL}"

                    print(labeled_line)
                    f.write(labeled_line + "\n")

                except:
                    # bỏ qua dòng lỗi
                    continue

    except KeyboardInterrupt:
        print("\n⏹️ Stopped manually.")

    finally:
        ser.close()
        print(f"\n✅ Done. File saved: {FILENAME}")
