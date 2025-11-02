import serial
import time

# === CẤU HÌNH ===
PORT = 'COM4'         # Cổng COM của Arduino
BAUD_RATE = 115200    # Giống như trong Serial.begin()
FILENAME = "4.csv"  # File CSV xuất ra
# ✅ Nhãn tư thế (có thể đổi: 1 = nằm ngửa, 2 = trái, 3 = phải, 4 = sấp)
LABEL = 4
DURATION = 60         # Thời gian ghi (giây)

# === MỞ KẾT NỐI COM ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Chờ board khởi động

print(f"✅ Ghi dữ liệu từ {PORT} vào '{FILENAME}' với nhãn LABEL = {LABEL}")
print("⏳ Nhấn Ctrl+C để dừng sớm...\n")

start_time = time.time()

with open(FILENAME, "w") as f:
    f.write("aX,aY,aZ,label\n")  # Header
    try:
        while time.time() - start_time < DURATION:
            line = ser.readline().decode("utf-8").strip()
            if line:
                labeled_line = f"{line},{LABEL}"  # ✅ Thêm nhãn từ biến LABEL
                print(labeled_line)
                f.write(labeled_line + "\n")
    except KeyboardInterrupt:
        print("\n⏹️ Ghi thủ công kết thúc.")
    finally:
        ser.close()
        print(f"\n✅ Ghi xong. File đã lưu tại: {FILENAME}")
