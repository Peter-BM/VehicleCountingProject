import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
ov_model = YOLO("yolov8s_openvino_model/")
cap = cv2.VideoCapture("/home/meneghini_/Desktop/faculdade/TCC/TCC1/imagensEVideos/RuaDia.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(100, 550), (1300, 550)]  # line or region points
classes_to_count = [2, 3, 5, 7]  # Car, Motorcycles, buses, trucks

# Video writer
# video_writer = cv2.VideoWriter("RuaDiaYs.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=False,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = ov_model.track(im0, persist=True, show=False, classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    # video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Informações do vídeo:
print(f'\n\nWidth: {w}\nHeight: {h}\nFPS: {fps}')

cap.release()
# video_writer.release()
cv2.destroyAllWindows()
