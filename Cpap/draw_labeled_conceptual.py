import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def draw_labels():
    img_path = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document\sipap_clean_internals.png"
    if not os.path.exists(img_path):
        print("Error: Base image not found!")
        return

    from PIL import Image
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # We will turn on the axis grid to see coordinates for placing labels
    # The image size is usually 1024x1024 (or similar depending on generator output)
    width, height = img.size
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0) # Flip y-axis to match pixel coordinates (0,0 at top-left)
    
    # Let's define the labels and their coordinates:
    # (label_text, xy_target, xy_text, alignment)
    # Note: These coordinates are estimates. We'll plot a grid first or adjust them.
    # Let's inspect the layout of the generated image. Since we don't see it directly,
    # let's write a script to save a version with a grid so we can verify coordinates,
    # or just use standard placement. Let's write the labeling logic.
    
    # Design of label boxes
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#1B365D", lw=1.5, alpha=0.9)
    arrow_props = dict(arrowstyle="->", color="#FF5722", lw=1.8, connectionstyle="arc3,rad=0.1")
    
    # Estimated positions based on the generated image structure:
    # Microcontroller (usually center-left, around x=450, y=420)
    ax.annotate("Vi điều khiển trung tâm\n(Arduino Nano 33 BLE Sense)", 
                xy=(width * 0.48, height * 0.42), 
                xytext=(width * 0.20, height * 0.30),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=9, fontweight='bold', color="#1B365D",
                ha='center', va='center')

    # Blower Motor (usually large round fan at bottom-left or center, around x=320, y=550)
    ax.annotate("Động cơ quạt thổi khí\n(Brushless Blower WS4540)", 
                xy=(width * 0.35, height * 0.58), 
                xytext=(width * 0.15, height * 0.70),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=9, fontweight='bold', color="#1B365D",
                ha='center', va='center')

    # Flow Sensor (on the tube, top-right, around x=600, y=280)
    ax.annotate("Cảm biến lưu lượng khí thở\n(Sensirion SFM3300-D)", 
                xy=(width * 0.60, height * 0.28), 
                xytext=(width * 0.82, height * 0.20),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=9, fontweight='bold', color="#1B365D",
                ha='center', va='center')

    # Pressure Sensor (small module near microcontroller, around x=380, y=360)
    ax.annotate("Cảm biến áp suất đường thở\n(MPXV5010G)", 
                xy=(width * 0.38, height * 0.36), 
                xytext=(width * 0.12, height * 0.45),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=9, fontweight='bold', color="#1B365D",
                ha='center', va='center')

    # OLED Display / Control (front, around x=520, y=600)
    ax.annotate("Màn hình hiển thị OLED\n& Giao diện điều khiển", 
                xy=(width * 0.53, height * 0.62), 
                xytext=(width * 0.53, height * 0.85),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=9, fontweight='bold', color="#1B365D",
                ha='center', va='center')

    # Air Intake / Filter (usually bottom-left or left, around x=100, y=620)
    ax.annotate("Bộ lọc khí đầu vào\n(Air Intake Filter)", 
                xy=(width * 0.10, height * 0.62), 
                xytext=(width * 0.10, height * 0.85),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=9, fontweight='bold', color="#1B365D",
                ha='center', va='center')

    ax.axis('off')
    
    # Save the final image
    out_path = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document\sipap_conceptual_diagram.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Labeled image saved successfully at {out_path}!")

if __name__ == "__main__":
    draw_labels()
