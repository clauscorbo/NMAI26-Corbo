"""Ensemble inference script for grocery product detection (ONNX).

Loads all ONNX models from model_weights/, runs each at its native resolution,
merges predictions with Weighted Box Fusion (WBF) + horizontal flip TTA.

Sandbox contract:
    python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model_weights"

CONF_THRESH = 0.001
IOU_THRESH = 0.5
WBF_IOU_THRESH = 0.60
MAX_DET = 300
ENABLE_TTA = True


# ── Preprocessing ──────────────────────────────────────────────


def letterbox(img, new_shape):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w = int(round(w * r))
    new_h = int(round(h * r))
    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def preprocess(img, use_fp16=False):
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    if use_fp16:
        img = img.astype(np.float16)
    return img[np.newaxis]


# ── Postprocessing ─────────────────────────────────────────────


def postprocess(output, ratio, pad, img_w, img_h,
                conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    preds = output[0].astype(np.float32).squeeze(0).T
    scores = preds[:, 4:]
    class_ids = scores.argmax(axis=1)
    max_scores = scores[np.arange(len(scores)), class_ids]

    mask = max_scores > conf_thresh
    if mask.sum() == 0:
        return []

    boxes_xywh = preds[mask, :4]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    dw, dh = pad
    x1 = np.clip((x1 - dw) / ratio, 0, img_w)
    y1 = np.clip((y1 - dh) / ratio, 0, img_h)
    x2 = np.clip((x2 - dw) / ratio, 0, img_w)
    y2 = np.clip((y2 - dh) / ratio, 0, img_h)

    boxes_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_nms, max_scores.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []

    indices = indices.flatten()[:MAX_DET * 3]
    return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
             float(max_scores[i]), int(class_ids[i])) for i in indices]


# ── Helpers ────────────────────────────────────────────────────


def get_imgsz(name):
    for part in Path(name).stem.split("_"):
        if part.isdigit() and int(part) >= 320:
            return int(part)
    return 1280


# ── Main ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model_files = sorted(MODEL_DIR.glob("*.onnx"))

    sessions = []
    for mf in model_files:
        sess = ort.InferenceSession(str(mf), providers=providers)
        imgsz = get_imgsz(mf.name)
        inp_name = sess.get_inputs()[0].name
        use_fp16 = "float16" in sess.get_inputs()[0].type
        sessions.append((sess, inp_name, imgsz, use_fp16))
        print(f"Loaded {mf.name}  imgsz={imgsz}  fp16={use_fp16}")

    is_ensemble = len(sessions) > 1
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in image_exts)

    predictions = []

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        if is_ensemble:
            det_lists = []
            for sess, inp_name, imgsz, use_fp16 in sessions:
                img_lb, ratio, pad = letterbox(img, imgsz)
                out = sess.run(None, {inp_name: preprocess(img_lb, use_fp16)})
                det_lists.append(postprocess(out, ratio, pad, img_w, img_h))

                if ENABLE_TTA:
                    img_f = img[:, ::-1].copy()
                    img_lb_f, r_f, p_f = letterbox(img_f, imgsz)
                    out_f = sess.run(None, {inp_name: preprocess(img_lb_f, use_fp16)})
                    dets_f = postprocess(out_f, r_f, p_f, img_w, img_h)
                    det_lists.append([(img_w-x2, y1, img_w-x1, y2, s, c)
                                      for x1, y1, x2, y2, s, c in dets_f])

            # Use ensemble_boxes WBF
            boxes_list, scores_list, labels_list = [], [], []
            for dets in det_lists:
                if not dets:
                    boxes_list.append(np.empty((0, 4)))
                    scores_list.append(np.empty(0))
                    labels_list.append(np.empty(0))
                else:
                    b = np.array([[x1/img_w, y1/img_h, x2/img_w, y2/img_h]
                                  for x1, y1, x2, y2, s, c in dets])
                    b = np.clip(b, 0, 1)
                    boxes_list.append(b)
                    scores_list.append(np.array([s for _, _, _, _, s, _ in dets]))
                    labels_list.append(np.array([c for _, _, _, _, _, c in dets]))

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                iou_thr=WBF_IOU_THRESH, skip_box_thr=0.0
            )

            dets = []
            for i in range(min(len(fused_scores), MAX_DET)):
                dets.append((
                    float(fused_boxes[i][0] * img_w),
                    float(fused_boxes[i][1] * img_h),
                    float(fused_boxes[i][2] * img_w),
                    float(fused_boxes[i][3] * img_h),
                    float(fused_scores[i]),
                    int(fused_labels[i])
                ))
        else:
            sess, inp_name, imgsz, use_fp16 = sessions[0]
            img_lb, ratio, pad = letterbox(img, imgsz)
            out = sess.run(None, {inp_name: preprocess(img_lb, use_fp16)})
            dets = postprocess(out, ratio, pad, img_w, img_h)

        for x1, y1, x2, y2, conf, cls_id in dets:
            predictions.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [round(max(0.0, x1), 2), round(max(0.0, y1), 2),
                         round(min(float(img_w), x2) - max(0.0, x1), 2),
                         round(min(float(img_h), y2) - max(0.0, y1), 2)],
                "score": round(conf, 4),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
