## HPCImageGeneration

### HPC Cluster Class Pair Image Generation Pipeline

---

## 🧰 How to Use

### 1. Build the Containers in Sandbox Mode:
```bash
apptainer build --force --sandbox model_test_sandbox model_test.def
apptainer build --sandbox image_detection_sandbox image_detection.def
```

### 2. Run on a GPU Node:
```bash
srun --partition=h100 --gres=gpu:1 --pty bash
```

### 3. Run the Image Generation Container:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True apptainer run --nv \
  --bind /home/s497288/lab-project/images:/images \
  --bind /home/s497288/lab-project/models:/models \
  ~/lab-project/model_test_sandbox \
  --output-dir /images \
  --model-path /models/FLUX.1-dev \
  --plan generation_plan.json \
  --steps 16 \
  --batch-size 16
```

### 4. Run the Object Detection Container:
```bash
apptainer run --nv \
  --bind /home/s497288/lab-project/images:/images \
  --bind /home/s497288/lab-project/results:/results \
  ~/lab-project/image_detection_sandbox \
  --images-dir /images \
  --results-dir /results \
  --model-path /yolo11x.pt
```

---

## 🛠️ Requirements

### 📁 Folder Structure
```bash
/home/s497288/lab-project/
├── images/            # Output directory for generated and processed images
├── models/            # Directory containing the FLUX.1-schnell model
│   └── FLUX.1-dev/    # Pretrained model directory for image generation
├── results/           # Output directory for object detection results
├── model_test.def     # Definition file for image generation container
├── image_detection.def# Definition file for object detection container
├── generation_plan.json # JSON file describing the class pair and image count plan
```

### 🤖 Models
- **FLUX.1-schnell**: Required for image generation. Place it in `/models/FLUX.1-dev`.
- **YOLOv11x (`yolo11x.pt`)**: Required for object detection. Place it in a readable path (e.g., `/yolo11x.pt`).

---

## ✨ Command-line Arguments Explained

### Image Generation Container (`model_test.py`)

```bash
apptainer run --nv \
  --bind /home/s497288/lab-project/images:/images \
  --bind /home/s497288/lab-project/models:/models \
  ~/lab-project/model_test_sandbox \
  --output-dir /images \
  --model-path /models/FLUX.1-dev \
  --plan generation_plan.json \
  --steps 16 \
  --batch-size 16
```

| Argument         | Description |
|------------------|-------------|
| `--output-dir`   | Path to store generated images. Must be writable. |
| `--model-path`   | Path to the directory containing the pretrained FLUX.1 model. |
| `--plan`         | JSON file that maps `"animal1|animal2"` pairs to image counts. |
| `--steps`        | Number of diffusion inference steps (default is 32). |
| `--batch-size`   | Number of images to generate per batch (default is 4). |

> 🔍 Example `generation_plan.json`:
```json
{
  "cat|dog": 16,
  "lion|tiger": 8
}
```

### Object Detection Container

```bash
apptainer run --nv \
  --bind /home/s497288/lab-project/images:/images \
  --bind /home/s497288/lab-project/results:/results \
  ~/lab-project/image_detection_sandbox \
  --images-dir /images \
  --results-dir /results \
  --model-path /yolo11x.pt
```

| Argument         | Description |
|------------------|-------------|
| `--images-dir`   | Directory of images to run object detection on (typically `/images`). |
| `--results-dir`  | Directory to save detection results (e.g., bounding boxes, labels). |
| `--model-path`   | Path to the pretrained YOLOv11x weights file. |

---
