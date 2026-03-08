import cv2
import logging

from camera_manager import CameraManager
from face_detector import FaceDetector
from filters.filtro_orejas import FiltroOrejas
from filters.mustach_filter import MustachFilter
from filters.lego_head import LegoHeadFilter

logging.getLogger("mediapipe").setLevel(logging.ERROR)

WINDOW_NAME = "Mallas"


def _build_filter(filter_key):
    if filter_key == "orejas":
        return FiltroOrejas()
    if filter_key == "mostacho":
        return MustachFilter(image_path="assets/mustache.png", object_width=100)
    if filter_key == "lego":
        return LegoHeadFilter()
    return None


def _safe_release(filter_obj):
    if filter_obj is None:
        return
    release_fn = getattr(filter_obj, "release", None)
    if callable(release_fn):
        try:
            release_fn()
        except Exception:
            pass


def _draw_menu(frame, current_key, show_menu, ui_state):
    if frame is None:
        ui_state["buttons"] = []
        return

    if not show_menu:
        ui_state["buttons"] = []
        return

    h, w = frame.shape[:2]

    buttons = [
        ("none", "Ninguno"),
        ("orejas", "Orejas"),
        ("mostacho", "Mostacho"),
        ("lego", "Lego"),
    ]

    bw = 102
    bh = 28
    gap = 8
    total_w = len(buttons) * bw + (len(buttons) - 1) * gap
    start_x = max(10, (w - total_w) // 2)
    y1 = max(10, h - bh - 12)

    panel_pad = 8
    panel_x1 = start_x - panel_pad
    panel_y1 = y1 - 30
    panel_x2 = start_x + total_w + panel_pad
    panel_y2 = y1 + bh + panel_pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    hint = "1/2/3/0 o click | H ocultar | Q salir"
    cv2.putText(frame, hint, (start_x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    ui_state["buttons"] = []
    for i, (key, label) in enumerate(buttons):
        x1 = start_x + i * (bw + gap)
        x2 = x1 + bw
        y2 = y1 + bh

        active = key == current_key
        fill = (40, 140, 40) if active else (50, 50, 50)
        border = (130, 255, 130) if active else (165, 165, 165)

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)
        cv2.putText(frame, label, (x1 + 8, y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        ui_state["buttons"].append({"key": key, "rect": (x1, y1, x2, y2)})


def _make_mouse_callback(ui_state):
    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for btn in ui_state.get("buttons", []):
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                ui_state["requested_filter"] = btn["key"]
                return

    return _on_mouse


cam = CameraManager(camera_index=1)
detector = FaceDetector(
    is_static=False,
    max_num_faces=1,
    use_landmarks=True,
    min_confidence_detection=0.5,
    min_confidence_tracking=0.5,
)

current_filter_key = "none"
selected_filter = None
show_menu = True

ui_state = {
    "buttons": [],
    "requested_filter": None,
}

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, _make_mouse_callback(ui_state))

try:
    while True:
        try:
            frame = cam.read_frame()
        except RuntimeError:
            continue

        if frame is None:
            continue

        landmarks = detector.detect(frame)
        output = detector.draw(frame)

        if output is None:
            output = frame.copy()

        if selected_filter and landmarks:
            filtered = selected_filter.apply(output, landmarks)
            if filtered is not None:
                output = filtered

        _draw_menu(output, current_filter_key, show_menu, ui_state)
        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("h"):
            show_menu = not show_menu

        next_key = ui_state.get("requested_filter")
        ui_state["requested_filter"] = None

        if key in (ord("0"), ord("n")):
            next_key = "none"
        elif key in (ord("1"), ord("o")):
            next_key = "orejas"
        elif key in (ord("2"), ord("m")):
            next_key = "mostacho"
        elif key in (ord("3"), ord("l")):
            next_key = "lego"

        if next_key is None or next_key == current_filter_key:
            continue

        _safe_release(selected_filter)
        selected_filter = _build_filter(next_key)
        current_filter_key = next_key

finally:
    _safe_release(selected_filter)
    cam.release()
    cv2.destroyAllWindows()
