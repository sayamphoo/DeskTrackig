import cv2
import torch
import queue
import threading
from datetime import datetime
import math
import time


class DeskTrackings:
    def __init__(self, human_model, desk_model, capture=0):
        self._video_capture = cv2.VideoCapture(capture)
        self.fps = self._video_capture.get(cv2.CAP_PROP_FPS)
        self._desk_model = desk_model
        self._human_model = human_model
        self._desks_detail = [
            # x,y,w,h,isHuman,point,timestamp,
        ]
        self._running = True
        self._table_queue = queue.Queue()
        self._human_queue = queue.Queue()
        self._lock = threading.Lock()

    def _diffs_time(self, time_old) -> tuple:
        current_time = datetime.now()
        diff_time = current_time - time_old
        diff_minutes = diff_time.total_seconds() // 60
        diff_seconds = diff_time.total_seconds() % 60
        diff_time_str = "{:02}:{:02}".format(int(diff_minutes), int(diff_seconds))
        return diff_time_str, current_time, diff_minutes, diff_seconds

    def _center_frame(self, x1, y1, x2, y2) -> tuple:
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        return x_center, y_center

    def _is_within_radius(self, x1, y1, x2, y2, radius=100):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance <= radius

    def _human_detect(self):
        while self._running:
            frame, desk = self._human_queue.get(block=True)
            if frame is None:
                continue

            x, y, w, h = map(int, (desk["x"], desk["y"], desk["w"], desk["h"]))
            if w <= 0 or h <= 0:
                continue

            cropped_frame = frame[y : y + h, x : x + w]
            results = self._human_model(cropped_frame)
            if results.xyxy[0] is None:
                continue

            is_human = len([res for res in results.xyxy[0] if res[5] == 0]) > 0

            with self._lock:
                if desk["isHuman"] != is_human:
                    desk["isHuman"] = is_human
                    desk["timestamp"] = datetime.now()

    def _desk_detect(self):
        while self._running:
            frame = self._table_queue.get(block=True)
            if frame is None:
                continue

            desks = self._desk_model(frame)

            for x1, y1, x2, y2, conf, _ in desks.xyxy[0].cpu().numpy():
                if conf > 0.55:
                    with self._lock:
                        center = self._center_frame(x1, y1, x2, y2)
                        found_existing_desk = False
                        for desk in self._desks_detail:
                            if self._is_within_radius(
                                desk["point"][0], desk["point"][1], *center
                            ):
                                desk.update(
                                    {
                                        "x": (x1 - 20),
                                        "y": (y1 - 20),
                                        "w": (x2 - x1 + 40),
                                        "h": (y2 - y1 + 40),
                                        "point": center,
                                        "isHuman": desk["isHuman"],
                                        "timestamp": desk["timestamp"],
                                    }
                                )
                                self._human_queue.put((frame, desk))
                                found_existing_desk = True
                                break

                        if not found_existing_desk:
                            new_desk = {
                                "x": (x1 - 20),
                                "y": (y1 - 20),
                                "w": (x2 - x1 + 40),
                                "h": (y2 - y1 + 40),
                                "point": center,
                                "isHuman": False,
                                "timestamp": datetime.now(),
                            }
                            self._desks_detail.append(new_desk)
                            self._human_queue.put((frame, new_desk))

    def draw_box(self, frame, desk):
        x, y, w, h = map(int, (desk["x"], desk["y"], desk["w"], desk["h"]))
        is_human = desk["isHuman"]
        diff_time_str, _, _, _ = self._diffs_time(desk["timestamp"])
        # BGR
        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (0, 255, 0) if is_human else (0, 0, 255), 2
        )

        cv2.putText(
            frame,
            f"{diff_time_str}-{'Active' if is_human else 'Inactive'}",
            (x + 10, y + h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if is_human else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def run(self):
        thread_table_detect = threading.Thread(target=self._desk_detect)
        thread_human_detect = threading.Thread(target=self._human_detect)

        thread_table_detect.start()
        thread_human_detect.start()
        start_time = time.time()

        while self._running:
            runtime = time.time()
            ret, frame = self._video_capture.read()
            if not ret:
                self._running = True
                break

            h, w = frame.shape[:2]
            scale = 1080 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 6:
                self._table_queue.put(frame)
                start_time = current_time  # รีเซ็ตเวลาเริ่มต้น

            for desk in self._desks_detail:
                self.draw_box(frame, desk)

            cv2.imshow("YOLOv5 Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False

            time.sleep((1.0 - (time.time() - runtime)) / self.fps)

        self._running = False
        thread_table_detect.join()
        thread_human_detect.join()
        self._video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    human_model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="./model/yolov5s.pt"
    )
    desk_model = torch.hub.load("ultralytics/yolov5", "custom", path="./model/best.pt")
    desk = DeskTrackings(
        human_model=human_model,
        desk_model=desk_model,
        capture="./validation/Video/IMG_9960.MOV",
    )
    desk.run()
