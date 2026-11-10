import os, sys, argparse, glob
import cv2
import torch
from PIL import Image, ImageDraw

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
YOLOV5_DIR = os.path.join(PROJECT_ROOT, 'yolov5')

# Make YOLOv5 modules importable
sys.path.insert(0, YOLOV5_DIR)

# --- Rebuild YOLOv5-LSE architecture from YAML ---
from models.yolo import Model
import yaml

with open(os.path.join(YOLOV5_DIR, 'models', 'yolov5s-lse.yaml')) as f:
    cfg = yaml.safe_load(f)

yolo_model = Model(cfg, ch=3, nc=1)

# --- Load pure state_dict ---
sd_path = os.path.join(WEIGHTS_DIR, 'yolov5_LSE_pure_state.pt')

if not os.path.exists(sd_path):
    print(f"❌ Missing file: {sd_path}")
    print(f"   Please copy 'yolov5_LSE_pure_state.pt' from Colab to weights/")
    sys.exit(1)

state = torch.load(sd_path, map_location='cpu', weights_only=True)
state = {f"model.{k}": v for k, v in state.items()}   # <-- ADD THIS LINE

if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']

yolo_model.load_state_dict(state, strict=True)
yolo_model.eval()
print("✅ YOLOv5-LSE loaded successfully!")

# --- Load CRNN ---
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code'))
import resume_setup

# --- Image processing ---
def process_image(image_path, output_dir=None):
    from utils.general import non_max_suppression
    from utils.augmentations import letterbox

    if output_dir is None:
                output_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(output_dir, exist_ok=True)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # --- Preprocess: resize to 640x640 keeping aspect ratio + letterbox padding ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_letter, ratio, pad = letterbox(img_rgb, 640, auto=False, scaleup=True)
    img_tensor = img_letter.transpose(2, 0, 1)                 # HWC -> CHW
    img_tensor = torch.from_numpy(img_tensor).float() / 255.0  # 0-1
    img_tensor = img_tensor.unsqueeze(0)                       # add batch dim

    # --- YOLO forward ---
    with torch.no_grad():
        raw = yolo_model(img_tensor)[0]                        # (B, N, 5+nc)

    # --- NMS ---
    nms_out = non_max_suppression(raw, conf_thres=0.15, iou_thres=0.45)
    dets = nms_out[0].cpu().numpy() if nms_out and nms_out[0] is not None else []

    # --- Draw results on original image ---
    out_img = img_bgr.copy()
    pil = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    plates = []
    for det in dets:
        x1, y1, x2, y2, conf = det[:5]
        # Undo letterbox padding/scale
        x1 = (x1 - pad[0]) / ratio[0]
        y1 = (y1 - pad[1]) / ratio[1]
        x2 = (x2 - pad[0]) / ratio[0]
        y2 = (y2 - pad[1]) / ratio[1]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)

        crop = out_img[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3]
        if crop.size == 0:
            continue
        text = resume_setup.recognize(crop)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        draw.text((x1 + 5, max(0, y1 - 25)), f"{text} ({conf:.2f})", fill=(0, 255, 0))
        plates.append({"text": text, "conf": float(conf)})

    out_path = os.path.join(output_dir, 'annotated_' + os.path.basename(image_path))
    pil.save(out_path)
    return plates, out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--folder")
    args = ap.parse_args()

    if args.image:
        plates, out = process_image(args.image)
        print(f"Plates: {plates}")
        print(f"Saved: {out}")
    elif args.folder:
        for img in sorted(glob.glob(os.path.join(args.folder, "*.jpg"))):
            plates, out = process_image(img)
            print(f"{os.path.basename(img)}: {[p['text'] for p in plates]}")
    else:
        print("Provide --image or --folder")