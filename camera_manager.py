import cv2 as cv

class CameraManager: # Esta clase levanta la cámara y lee los frames de video. Es una clase simple que encapsula la funcionalidad de la cámara.
    def __init__(self, camera_index=1):
        self.cap = cv.VideoCapture(camera_index) # Levanta la cámara. El índice 0 generalmente se refiere a la cámara predeterminada del sistema.

    def read_frame(self):
        ret, frame = self.cap.read() # Lee un frame de la cámara. 'ret' es un booleano que indica si la lectura fue exitosa, y 'frame' es la imagen capturada.
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self):
        """Libera la cámara y destruye las ventanas de OpenCV."""
        # Intentar liberar el recurso de la cámara si existe y está abierto
        if hasattr(self, "cap") and self.cap is not None:
            try:
                if hasattr(self.cap, "isOpened") and self.cap.isOpened():
                    self.cap.release()
                else:
                    # intentar release() aunque isOpened() no exista o devuelva False
                    try:
                        self.cap.release()
                    except Exception:
                        pass
            except Exception:
                try:
                    self.cap.release()
                except Exception:
                    pass
            finally:
                self.cap = None

        # Cerrar todas las ventanas creadas por OpenCV
        try:
            cv.destroyAllWindows()
        except Exception:
            pass