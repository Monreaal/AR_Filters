import cv2
import mediapipe
import logging
from camera_manager import CameraManager
from face_detector import FaceDetector
from filters.filtro_orejas import FiltroOrejas
from filters.mustach_filter import MustachFilter

logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Solo muestra errores, no warnings

def elegir_filtro():
    print("Selecciona filtro:")
    print("  o / orejas   -> Filtro de orejas")
    print("  m / mostacho -> Filtro de mostacho")
    print("  none         -> Ninguno")
    opcion = input("Opción: ").strip().lower()
    if opcion in ("o", "orejas", "oreja"):
        return FiltroOrejas()
    if opcion in ("m", "mostacho", "bigote", "mustache"):
        # ruta relativa al archivo de bigote
        path = "assets/mustache.png"
        # ancho base puede ajustarse; 100 parece un tamaño razonable
        return MustachFilter(image_path=path, object_width=100)
    return None


cam = CameraManager(camera_index=1)
detector = FaceDetector(is_static=False, max_num_faces=1, use_landmarks=True, min_confidence_detection=0.5, min_confidence_tracking=0.5)

selected_filter = elegir_filtro()

try:
    while True:
        
        frame = cam.read_frame()

        # Obtener landmarks y una imagen con la malla dibujada
        landmarks = detector.detect(frame)
        output = detector.draw(frame)

        # Aplicar filtro si fue seleccionado
        if selected_filter and landmarks:
            selected_filter.apply(output, landmarks)

        cv2.imshow("Mallas", output)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
finally:
    cam.release()