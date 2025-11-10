import matplotlib.pyplot as plt

# Dữ liệu
labels = [
    'Thu thập, làm sạch dữ liệu', 'Đánh nhãn dữ liệu', 'Bổ sung dữ liệu',
    'Phân tích, trích xuất đặc trưng', 'Chọn tham số và huấn luyện', 'Đánh giá, điều chỉnh mô hình',
    'Chuyển đổi, nén mô hình', 'Triển khai trên biên', 'Vận hành'
]
sizes = [50, 2,  1, 20, 5, 5, 5, 10, 2]
colors = [
    '#4F81BD', '#C0504D', '#9BBB59', '#8064A2', '#4BACC6',
    '#F79646', '#7F7F7F', '#BFBFBF', '#92D050'
]

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
wedges, texts, autotexts = ax.pie(
    sizes,
    colors=colors,
    startangle=70,
    wedgeprops={'edgecolor': 'white'},
    autopct='%1.0f%%',
    pctdistance=0.8,       # đặt % gần giữa lát
    labeldistance=1.1      # đưa nhãn ra ngoài
)

# Căn chỉnh nhãn và % cho dễ đọc
plt.setp(autotexts, size=8, weight='bold', color='white')
plt.setp(texts, size=9)

# Thêm tiêu đề
ax.set_title('Tỷ lệ công việc trong bài toán học máy phân loại tư thế ngủ',
             fontsize=13, fontweight='semibold', pad=20)

# Thêm legend (chú thích) bên phải
ax.legend(
    wedges,
    labels,
    title="Các công đoạn",
    loc="center left",
    bbox_to_anchor=(0.8, 0, 0.3, 0.7),
    fontsize=9
)

# Cân đối hình và lưu
ax.axis('equal')
plt.tight_layout()
plt.savefig("piechart_xuly_dulieu_legend.png", dpi=300, bbox_inches='tight')
plt.show()
