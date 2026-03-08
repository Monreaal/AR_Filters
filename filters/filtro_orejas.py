import cv2 as cv
import numpy as np
from .base_filter import BaseFilter 

class FiltroOrejas(BaseFilter):
    def apply(self, frame, landmarks):
        if landmarks and len(landmarks) > 0:
            face = landmarks[0]
            h, w = frame.shape[:2]

            # Índices aproximados para los lados de la cara (zona orejas)
            left_points = [234, 93]
            right_points = [454, 323]

            # ---- OREJA IZQUIERDA ----
            lx1 = int(face.landmark[left_points[0]].x * w)
            ly1 = int(face.landmark[left_points[0]].y * h)

            lx2 = int(face.landmark[left_points[1]].x * w)
            ly2 = int(face.landmark[left_points[1]].y * h)

            # Punto externo (para formar triángulo hacia afuera)
            lx3 = lx1 - 40
            ly3 = ly1 - 60

            left_triangle = np.array([
                [lx1, ly1],
                [lx2, ly2],
                [lx3, ly3]
            ], np.int32)

            cv.fillPoly(frame, [left_triangle], (0, 255, 0))

            # ---- OREJA DERECHA ----
            rx1 = int(face.landmark[right_points[0]].x * w)
            ry1 = int(face.landmark[right_points[0]].y * h)

            rx2 = int(face.landmark[right_points[1]].x * w)
            ry2 = int(face.landmark[right_points[1]].y * h)

            # Punto externo
            rx3 = rx1 + 40
            ry3 = ry1 - 60

            right_triangle = np.array([
                [rx1, ry1],
                [rx2, ry2],
                [rx3, ry3]
            ], np.int32)

            cv.fillPoly(frame, [right_triangle], (0, 255, 0))