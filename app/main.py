import onnxruntime as ort
import numpy as np
import cv2
import time

MODEL_PATH = "/app/app/yolov8n.onnx"
VIDEO_PATH = "/app/app/traffic.avi"
INPUT_SIZE = 320


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


def run_video(session):
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video")
        return

    input_name = session.get_inputs()[0].name

    total_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        total_frames += 1

    end_time = time.time()
    cap.release()

    total_time = end_time - start_time
    fps = total_frames / total_time

    print("Total frames:", total_frames)
    print("Total time (s):", round(total_time, 2))
    print("Average FPS:", round(fps, 2))


if __name__ == "__main__":
    session = create_session()
    run_video(session)