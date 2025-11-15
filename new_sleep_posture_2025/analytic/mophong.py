import matplotlib.pyplot as plt
import numpy as np
import itertools

# Ma trận nhầm lẫn
cm = np.array([
    [5495,  190,   86,    4],
    [0,    1618, 0,   0],
    [18,    0,    4852, 0],
    [0,    0,    11,  475]
])

# Nhãn tiếng Việt
classes = ["Ngửa", "Nghiêng trái", "Nghiêng phải", "Sấp"]

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Ma trận nhầm lẫn", fontsize=16)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

# Ghi số lên từng ô
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j, i, format(cm[i, j], "d"),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
        fontsize=12
    )

plt.ylabel('Nhãn thật', fontsize=13)
plt.xlabel('Nhãn dự đoán', fontsize=13)
plt.tight_layout()
plt.show()
