import time
from ultralytics import YOLO
import cv2
import numpy as np
import json  # Import JSON module


def select_points(event, x, y, flags, param):
    global points, selecting
    if event == cv2.EVENT_LBUTTONDOWN and selecting:
        if len(points) < 4:
            points.append((x, y))
        else:
            selecting = False


def calculate_centroid(pts):
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    return int(np.mean(x)), int(np.mean(y))


def is_ball_under_polygon(ball_center, polygon):
    sorted_points = sorted(polygon, key=lambda point: point[1], reverse=True)
    bottom_point1, bottom_point2 = sorted_points[0], sorted_points[1]

    dx = bottom_point2[0] - bottom_point1[0]
    dy = bottom_point2[1] - bottom_point1[1]

    if dx == 0:
        return ball_center[0] == bottom_point1[0] and ball_center[1] > bottom_point1[1]

    slope = dy / dx
    intercept = bottom_point1[1] - slope * bottom_point1[0]
    line_y_at_ball_x = slope * ball_center[0] + intercept
    return ball_center[1] > line_y_at_ball_x


def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


if __name__ == '__main__':
    model = YOLO("best.pt")
    video_path = "videos/vid4_hoograven.mov"
    output_path = "output/output.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class_names = model.names
    ball_class_id = 0
    confidence_threshold = 0.25
    selecting = True
    cv2.namedWindow("Ball Trajectory")
    cv2.setMouseCallback("Ball Trajectory", select_points)

    points = []
    trajectory_points = []
    FOOTBALL_DIAMETER_CM = 22
    do_trajectory = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_to_process = frame.copy()

        if selecting:
            for point in points:
                cv2.circle(frame, point, 10, (0, 0, 255), -1)

            if len(points) == 4:
                centroid = calculate_centroid(points)
                selecting = False
        else:
            results = model(img_to_process)

            cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 255), thickness=2)

            json_output = {"frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)), "objects": []}

            for result in results:
                if len(result.boxes) > 0:
                    ball_boxes = [box for box in result.boxes if
                                  int(box.cls) == ball_class_id and box.conf >= confidence_threshold]
                    for box in ball_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy())
                        class_id = int(box.cls.cpu().numpy())
                        class_name = class_names[class_id]

                        json_output["objects"].append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name
                        })

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print JSON output to the terminal
            print(json.dumps(json_output, indent=4))

        out.write(frame)
        cv2.imshow("Ball Trajectory", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
