from .base_filter import BaseFilter
import numpy as np
import cv2 as cv


class MustachFilter(BaseFilter):
    def __init__(self, image_path: str = None, object_width: int = None, offset: int = 0):
        """Carga la imagen del bigote y configura el ancho.

        :param image_path: ruta de la imagen con transparencia (canal alfa).
        :param object_width: ancho deseado del bigote en píxeles. Si se omite,
                             se utiliza el ancho de la imagen cargada.
        :param offset: desplazamiento vertical adicional al colocar el bigote.
        """

        self.overlay_rgba = None  # Imagen del bigote con canal alfa
        self.object_width = object_width  # Ancho objetivo para el bigote (en píxeles)
        self.offset = offset

        if not image_path:  # Si no se proporcionó una ruta de imagen, no se carga ningún filtro
            return

        img = cv.imread(image_path, cv.IMREAD_UNCHANGED)  # Cargar la imagen con canal alfa (si existe)
        if img is None:
            return

        overlay = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)  # Convertir de BGRA a RGBA

        # recortar cualquier área totalmente transparente alrededor del bigote
        alpha = overlay[..., 3]
        ys, xs = np.where(alpha > 0)
        if ys.size > 0 and xs.size > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            overlay = overlay[y1:y2+1, x1:x2+1]
        self.overlay_rgba = overlay

        # Si no se pasó un ancho objetivo, usar el ancho de la imagen recortada
        if self.object_width is None:
            _, w = overlay.shape[:2]
            self.object_width = w

    def landmark_xtoy(self, landmark, image_shape):
        """Convierte coordenadas normalizadas a píxeles."""
        h, w = image_shape[:2]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        return x, y

    def apply(self, frame, landmarks):
        # si no se cargó la imagen o no hay landmarks no hacemos nada
        if self.overlay_rgba is None or not landmarks:
            return frame

        nose = landmarks[0].landmark[1]  # punta de la nariz
        xnose, ynose = self.landmark_xtoy(nose, frame.shape)
        oh, ow = self.overlay_rgba.shape[:2]
        scale = self.object_width / ow

        new_oh = int(oh * scale)
        new_ow = int(ow * scale)

        resized = cv.resize(self.overlay_rgba, (new_ow, new_oh), interpolation=cv.INTER_AREA)

        # centrar horizontalmente respecto a la nariz
        x = xnose - new_ow // 2
        # para la vertical usamos la mitad de la altura de la imagen redimensionada
        # (no el ancho, ya que el bigote puede ser más ancho que alto).
        # esto ubica el bigote justo debajo de la punta de la nariz.
        y = ynose + new_oh // 2 + self.offset

        return self.overlay_rgba_on_bgr(frame, resized, x, y)

    def overlay_rgba_on_bgr(self, frame, rgba, x, y):
        h, w = frame.shape[:2]
        oh, ow = rgba.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + ow, w)
        y2 = min(y + oh, h)

        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        crop = rgba[oy1:oy2, ox1:ox2]
        if crop.size == 0:
            return frame

        rgb = crop[..., :3]
        roi = frame[y1:y2, x1:x2]

        # canal alfa normalizado y expandido para permitir broadcast con RGB
        alpha = crop[..., 3] / 255.0
        if alpha.ndim == 2:
            alpha = alpha[..., None]

        # las tres matrices deben coincidir en forma
        if roi.shape != rgb.shape:
            # en caso de desajuste, devolver frame sin modificación
            return frame

        blended = alpha * rgb + (1 - alpha) * roi

        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame
