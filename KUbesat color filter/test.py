import cv2
import numpy as np
import os
import argparse

def filter_color(image_path, lower_bound, upper_bound, output_path=None, show=False):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"[ERROR] Could not read image from {image_path}")

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Default output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_filtered{ext}"

    # Save result
    success = cv2.imwrite(output_path, result)
    if not success:
        raise IOError(f"[ERROR] Could not save image to {output_path}")
    print(f"[INFO] Filtered image saved to {output_path}")

    if show:
        cv2.imshow("Original", image)
        cv2.imshow("Filtered", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result, mask  # Useful for chaining filters or visualization

def main():
    parser = argparse.ArgumentParser(description="Filter image by HSV color range.")
    parser.add_argument("--input", required=True, help="Path to input image.")
    parser.add_argument("--output", help="Path to save filtered image.")
    parser.add_argument("--show", action="store_true", help="Display image result.")
    args = parser.parse_args()

    # Example: green HSV range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    filter_color(args.input, lower_green, upper_green, output_path=args.output, show=args.show)

if __name__ == "__main__":
    main()
