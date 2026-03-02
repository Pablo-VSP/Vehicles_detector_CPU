import onnxruntime as ort
import numpy as np
import cv2
import json
import time

MODEL_PATH = "yolov8n.onnx"
VIDEO_PATH = "traffic.avi"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.4

# Clases COCO relevantes
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def create_session():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )

    print("Providers:", session.get_providers())
    return session


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, frame_id, original_shape):
    preds = output[0]          # (1, 84, 2100)
    preds = np.squeeze(preds)  # (84, 2100)
    preds = preds.T            # (2100, 84)

    boxes = []
    scores = []
    class_ids = []

    img_h, img_w = original_shape

    for det in preds:
        x, y, w, h = det[0:4]
        class_scores = det[4:]

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < CONF_THRESHOLD:
            continue

        if class_id not in VEHICLE_CLASSES:
            continue

        # Convert YOLO center format → x1,y1,x2,y2
        x1 = (x - w / 2) * img_w / INPUT_SIZE
        y1 = (y - h / 2) * img_h / INPUT_SIZE
        x2 = (x + w / 2) * img_w / INPUT_SIZE
        y2 = (y + h / 2) * img_h / INPUT_SIZE

        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=CONF_THRESHOLD,
        nms_threshold=0.45
    )

    frame_result = {
        "frame": frame_id,
        "vehicles": []
    }

    counts = {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0,
        "bicycle": 0
    }

    if len(indices) > 0:
        for i in indices.flatten():
            label = VEHICLE_CLASSES[class_ids[i]]
            counts[label] += 1

            frame_result["vehicles"].append({
                "type": label,
                "confidence": scores[i]
            })

    frame_result["counts"] = counts
    return frame_result


def run_video(session):
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    input_name = session.get_inputs()[0].name
    frame_id = 0
    results = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        frame_result = postprocess(outputs, frame_id, frame.shape[:2])
        results.append(frame_result)

        print(json.dumps(frame_result))

        frame_id += 1

    end_time = time.time()
    cap.release()

    total_time = end_time - start_time
    fps = frame_id / total_time

    print("\n===== SUMMARY =====")
    print("Total frames:", frame_id)
    print("Total time (s):", round(total_time, 2))
    print("Average FPS:", round(fps, 2))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    session = create_session()
    run_video(session)