import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_FPN_Weights, FastRCNNPredictor

def preprocess(frame, size, device):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    resized = cv2.resize(image, (size, size))
    norm = resized / 255.
    norm -= np.array([0.485, 0.456, 0.406])
    norm /= np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).float().unsqueeze(0).to(device)
    return tensor, height, width

def load_model(checkpoint_path, num_classes, device):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_threshold = 0.6
    size = 416
    checkpoint_path = "trained_models/last_model.pt"
    categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']

    model = load_model(checkpoint_path, len(categories), device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor, h, w = preprocess(frame, size, device)

        with torch.no_grad():
            outputs = model(image_tensor)

        for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
            if score > conf_threshold:
                x1, y1, x2, y2 = box
                x1 = int(x1 / size * w)
                y1 = int(y1 / size * h)
                x2 = int(x2 / size * w)
                y2 = int(y2 / size * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                text = f"{categories[label]} {score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Faster R-CNN Webcam Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
