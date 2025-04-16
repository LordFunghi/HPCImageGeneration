#!/usr/bin/env python
import os
import cv2
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt

def annotate_image(image, animal1_box, animal2_box, threshold_line, outcome):
    """
    Returns the original image without visual annotations.
    Bounding boxes and threshold lines are intentionally omitted.
    """
    return image.copy()

def transform_box(box, scale):
    """
    Scales a bounding box (list of 4 numbers) by a uniform scale factor.
    """
    return [coord * scale for coord in box]

def zoom_in(image, delta):
    """
    Crops the bottom 'delta' pixels from the image and then resizes the cropped image
    back to the original size to effectively zoom in.
    """
    h, w = image.shape[:2]
    cropped = image[0 : h - int(delta), :]
    scale_factor = h / (h - delta)
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed, scale_factor

def process_image(image_path, model, conf_threshold, threshold_from_bottom, salvage_margin):
    """
    Processes an image:
      - Detects objects using YOLO and assigns Animal1 (upper) and Animal2 (lower) based on vertical center.
      - Computes the threshold line (from the bottom of the image).
      - If Animal2 intrudes above the threshold, and if intrusion is within salvage margin, outcome is "salvaged".
      - For "salvaged", the image is zoomed in (by cropping the intrusion portion) so that the animal no longer crosses.
      - Returns the annotated image and outcome ("passed", "failed", or "salvaged").
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    h, w = image.shape[:2]
    threshold_line = h - threshold_from_bottom

    results = model(image, conf=conf_threshold)
    if not results:
        print(f"No results for {image_path}")
        return None

    result = results[0]
    if result.boxes is None or len(result.boxes) < 2:
        print(f"Warning: less than 2 detections in {image_path}")
        return None

    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy.numpy()
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) / 2.0
        detections.append({"box": box, "center_y": center_y})
    detections = sorted(detections, key=lambda d: d["center_y"])
    
    animal1_box = detections[0]["box"]
    animal2_box = detections[1]["box"]
    animal2_top = animal2_box[1]

    if animal2_top < threshold_line:
        delta = threshold_line - animal2_top
        if delta <= salvage_margin:
            outcome = "salvaged"
        else:
            outcome = "failed"
    else:
        outcome = "passed"

    if outcome == "salvaged":
        zoomed_image, scale_factor = zoom_in(image, delta)
        new_animal1_box = transform_box(animal1_box, scale_factor)
        new_animal2_box = transform_box(animal2_box, scale_factor)
        new_threshold_line = h - threshold_from_bottom
        annotated = annotate_image(zoomed_image, new_animal1_box, new_animal2_box, new_threshold_line, outcome)
    else:
        annotated = annotate_image(image, animal1_box, animal2_box, threshold_line, outcome)
    
    return annotated, outcome

def main():
    parser = argparse.ArgumentParser(description="Detect if Animal2 crosses a configurable threshold using YOLO.")
    parser.add_argument("--images", type=str, required=True, help="Directory with input images")
    parser.add_argument("--results", type=str, required=True, help="Directory to save annotated images and the summary plot")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Path to your YOLO model (.pt file)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--threshold-from-bottom", type=int, required=True,
                        help="Threshold distance (pixels) from the bottom of the image (threshold_line = image_height - this value)")
    parser.add_argument("--salvage-margin", type=int, default=0, help="Salvage margin in pixels for borderline cases")
    args = parser.parse_args()
    
    os.makedirs(args.results, exist_ok=True)
    
    print(f"📥 Loading model from {args.model} ...")
    model = YOLO(args.model)
    
    image_files = [os.path.join(args.images, f) for f in os.listdir(args.images)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("❌ No images found in input directory.")
        return

    outcomes_by_pair = {}
    passed_count = 0
    failed_count = 0
    salvaged_count = 0
    
    for image_path in image_files:
        fname = os.path.basename(image_path)
        print(f"⏳ Processing {fname} ...", end=" ")
        proc_result = process_image(image_path, model, args.conf, args.threshold_from_bottom, args.salvage_margin)
        if proc_result is None:
            print("❌ Error processing image.")
            continue
        annotated, outcome = proc_result
        
        parts = fname.split('_')
        if len(parts) >= 3:
            class_pair = f"{parts[0]}_{parts[1]}"
        else:
            class_pair = "unknown"
        
        outcomes_by_pair.setdefault(class_pair, {"passed": 0, "failed": 0, "salvaged": 0})
        outcomes_by_pair[class_pair][outcome] += 1
        
        if outcome == "passed":
            passed_count += 1
            print("✅ Passed")
        elif outcome == "failed":
            failed_count += 1
            print("❌ Failed")
        elif outcome == "salvaged":
            salvaged_count += 1
            print("🟡 Salvaged")
        
        output_path = os.path.join(args.results, fname)
        cv2.imwrite(output_path, annotated)
    
    print("\nSummary:")
    print(f"✅ Passed:  {passed_count}")
    print(f"❌ Failed:  {failed_count}")
    print(f"🟡 Salvaged: {salvaged_count}")
    
    if outcomes_by_pair:
        class_pairs = sorted(outcomes_by_pair.keys())
        passed_vals = [outcomes_by_pair[k]["passed"] for k in class_pairs]
        salvaged_vals = [outcomes_by_pair[k]["salvaged"] for k in class_pairs]
        failed_vals = [outcomes_by_pair[k]["failed"] for k in class_pairs]
        
        x = range(len(class_pairs))
        plt.figure(figsize=(10, 6))
        p1 = plt.bar(x, passed_vals, color='green', label="Passed")
        p2 = plt.bar(x, salvaged_vals, bottom=passed_vals, color='yellow', label="Salvaged")
        bottom_vals = [p + s for p, s in zip(passed_vals, salvaged_vals)]
        p3 = plt.bar(x, failed_vals, bottom=bottom_vals, color='red', label="Failed")
        plt.xticks(x, class_pairs, rotation=45, ha='right')
        plt.xlabel("Class Pair")
        plt.ylabel("Count")
        plt.title("Outcome Counts per Class Pair")
        plt.legend()
        
        total = passed_count + failed_count + salvaged_count
        if total > 0:
            percent_passed = (passed_count / total) * 100
            percent_failed = (failed_count / total) * 100
            percent_salvaged = (salvaged_count / total) * 100
            summary_text = (f"Total: {total}\n"
                            f"Passed: {passed_count} ({percent_passed:.1f}%)\n"
                            f"Failed: {failed_count} ({percent_failed:.1f}%)\n"
                            f"Salvaged: {salvaged_count} ({percent_salvaged:.1f}%)")
            plt.figtext(0.75, 0.75, summary_text, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plot_path = os.path.join(args.results, "summary_plot.png")
        plt.savefig(plot_path)
        print(f"📊 Summary plot saved to {plot_path}")
        plt.show()

if __name__ == "__main__":
    main()
