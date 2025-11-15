import pandas as pd
from datetime import datetime

df = pd.read_excel("../datas/data_khong_nam.xlsx")


def convert_date(d):
    # nếu pandas đã đọc là datetime
    if isinstance(d, datetime):
        dt = d
    else:
        try:
            dt = datetime.strptime(str(d), "%m/%d/%Y")
        except:
            dt = pd.to_datetime(d)
    return dt.strftime("%a %b %d %Y %H:%M:%S GMT+0700 (Indochina Time)")


def parse_xyz(val):
    parts = [p.strip() for p in str(val).split(",")]
    return float(parts[0]), float(parts[1]), float(parts[2])


rows = []

for _, row in df.iterrows():
    customer = row["Tên"]
    activity = 0
    createdAt = convert_date(row["Ngày tạo"])
    x, y, z = parse_xyz(row["Giá trị"])

    rows.append([customer, activity, createdAt, x, y, z])

out_df = pd.DataFrame(
    rows, columns=["customer", "activity", "createdAt", "x", "y", "z"])
out_df.to_csv("converted.csv", index=False)

print("Đã tạo file converted.csv")
