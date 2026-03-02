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


def postprocess(output, frame_id):
    predictions = output[0][0]  # YOLOv8 shape: (num_boxes, 85)

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

    for det in predictions:
        conf = det[4]
        if conf < CONF_THRESHOLD:
            continue

        class_id = np.argmax(det[5:])
        score = det[5 + class_id]

        if class_id in VEHICLE_CLASSES and score > CONF_THRESHOLD:
            label = VEHICLE_CLASSES[class_id]
            counts[label] += 1

            frame_result["vehicles"].append({
                "type": label,
                "confidence": float(score)
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

        frame_result = postprocess(outputs, frame_id)
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