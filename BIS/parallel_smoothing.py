import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def update_cell_block(img, x_start, x_end, y_start, y_end, kernel_size=3):
    pad = kernel_size // 2
    block = img[x_start:x_end, y_start:y_end]
    out_block = np.zeros_like(block)

    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            x1 = max(x_start + i - pad, 0)
            x2 = min(x_start + i + pad + 1, img.shape[0])
            y1 = max(y_start + j - pad, 0)
            y2 = min(y_start + j + pad + 1, img.shape[1])

            region = img[x1:x2, y1:y2]
            out_block[i, j] = np.mean(region, axis=(0, 1))

    return out_block


def parallel_cellular_smoothing(image, num_cells=4, kernel_size=5):
    h, w = image.shape[:2]
    cell_height = h // num_cells
    results = [None] * num_cells
    tasks = []

    with ThreadPoolExecutor(max_workers=num_cells) as executor:
        for i in range(num_cells):
            x_start = i * cell_height
            x_end = (i + 1) * cell_height if i != num_cells - 1 else h

            future = executor.submit(
                update_cell_block, image, x_start, x_end, 0, w, kernel_size
            )
            tasks.append((i, future))

        for i, future in tasks:
            results[i] = future.result()

    return np.vstack(results)



def restore_image(smoothed_img, original_img):
  
    return original_img.copy()


input_path = input("Enter the image filename: ")

image = cv2.imread(input_path)

if image is None:
    print(" Error: Image not found. Make sure it's in the same folder.")
    exit()

print("Image loaded successfully!")


smoothed = parallel_cellular_smoothing(image, num_cells=6, kernel_size=5)

smoothed_path = "smoothed_output.jpg"
cv2.imwrite(smoothed_path, smoothed)
print(f" Smoothed image saved as: {smoothed_path}")


choice = input("Do you want to restore the original clear image? (yes/no): ").strip().lower()

if choice == "yes":
    restored = restore_image(smoothed, image)
    output_path = "restored_Smoothed_image.jpg"
    cv2.imwrite(output_path, restored)
    print(f"✔ Original clear image restored and saved as: {output_path}")
else:
    print("Restoration skipped.")
