import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import trimesh


class LegoHeadFilter:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        # Load 3D model
        self.model_path = Path(__file__).parent.parent / "assets" / "lego_head.glb"
        if self.model_path.exists():
            self.mesh = trimesh.load(str(self.model_path), force='mesh')
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Simplify mesh to reduce drawing cost (si tiene muchas caras)
        if len(self.mesh.faces) > 2000:
            try:
                self.mesh = self.mesh.simplify_quadratic_decimation(2000)
            except Exception:
                pass

        # Precompute mesh vertices/faces para proyección rápida
        self.mesh_vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.mesh_faces = np.asarray(self.mesh.faces, dtype=np.int32)

        # Center mesh at the origin and normalize size (unit sphere)
        centroid = self.mesh_vertices.mean(axis=0)
        self.mesh_vertices -= centroid
        max_dim = np.max(np.linalg.norm(self.mesh_vertices, axis=1))
        if max_dim > 0:
            self.mesh_vertices /= max_dim

        # Calcular altura de la malla en el eje Y (para escalar según la altura de la cara)
        self.mesh_y_min = float(np.min(self.mesh_vertices[:, 1]))
        self.mesh_y_max = float(np.max(self.mesh_vertices[:, 1]))
        self.mesh_height = self.mesh_y_max - self.mesh_y_min

        # Ajustar (aproximadamente) para que la punta de la nariz quede en el origen.
        # Esto hace que solvePnP (que usa como referencia el punto de la nariz) se alinee mejor.
        z = self.mesh_vertices[:, 2]
        z_mean = z.mean()
        if (z.max() - z_mean) > (z_mean - z.min()):
            # el eje +Z apunta hacia adelante; usar el vértice más adelantado
            tip_idx = int(np.argmax(z))
        else:
            # el eje -Z apunta hacia adelante
            tip_idx = int(np.argmin(z))
        tip = self.mesh_vertices[tip_idx]
        self.mesh_vertices -= tip

        # Puntos clave de referencia para solvePnP (Medio de cara)
        # Estos índices usan el esquema de MediaPipe FaceMesh.
        self._face_mesh_idxs = {
            'nose_tip': 1,
            'chin': 152,
            'left_eye_outer': 33,
            'right_eye_outer': 263,
            'left_mouth': 61,
            'right_mouth': 291
        }

        # Modelo 3D aproximado (en mm) para solvePnP.
        # Estos valores son los usados en el ejemplo de OpenCV.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # nariz
            (0.0, -63.6, -12.5),       # barbilla
            (-43.3, 32.7, -26.0),      # ojo izquierdo
            (43.3, 32.7, -26.0),       # ojo derecho
            (-28.9, -28.9, -24.1),     # comisura izquierda
            (28.9, -28.9, -24.1)       # comisura derecha
        ], dtype=np.float32)

        # Initialize camera matrix (adjust based on your camera)
        self.focal_length = 1000
        self.h, self.w = 480, 640
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.w / 2],
            [0, self.focal_length, self.h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1))

    def _project_mesh(self, frame, rvec, tvec, scale, offset2d=(0, 0)):
        """Proyecta el mesh en la imagen usando pose (rvec/tvec) y escala.

        Dibuja la malla rellena para dar sensación de sólido, y luego añade
        un wireframe para marcar los bordes.

        offset2d: corrección en pixeles para alinear el modelo con los landmarks.
        """
        # scale: factor relativo para adaptar tamaño de la malla al rostro
        verts = self.mesh_vertices * scale

        # desplazar el casco hacia arriba en el espacio 3D
        verts[:, 1] += scale * 120
        verts[:, 2] -= scale * 40

        # Rotar y trasladar
        R, _ = cv2.Rodrigues(rvec)
        verts_world = verts @ R.T + tvec.reshape(1, 3)

        # Proyectar 3D -> 2D
        projected, _ = cv2.projectPoints(verts_world, np.zeros((3, 1)), np.zeros((3, 1)), self.camera_matrix, self.dist_coeffs)
        projected = projected.reshape(-1, 2).astype(np.int32)

        # Aplicar offset en 2D (para centrar en la cara)
        if offset2d != (0, 0):
            projected = projected + np.array(offset2d, dtype=np.int32)

        # Crear máscara a partir de la proyección (llenar todos los triángulos)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        faces = projected[self.mesh_faces]

        # Evitar errores de fillPoly si el conjunto es demasiado grande
        try:
            cv2.fillPoly(mask, faces, 255)
        except Exception:
            for face in faces:
                cv2.fillPoly(mask, [face], 255)

        # Generar overlay de color sólido a partir de la máscara
        color_fill = (80, 190, 240)
        overlay = np.zeros_like(frame)
        overlay[:, :] = color_fill
        overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

        # Mezclar overlay con la imagen original usando alpha
        alpha_fill = 0.85
        cv2.addWeighted(overlay, alpha_fill, frame, 1 - alpha_fill, 0, frame)

        # Dibujar bordes (wireframe) encima del relleno, para claridad
        edge_color = (15, 75, 110)
        for face in faces:
            cv2.polylines(frame, [face], isClosed=True, color=edge_color, thickness=1)

        return frame

    def _overlay_rgba_on_bgr(self, frame, rgba, x, y):
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
        alpha = crop[..., 3:] / 255.0

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = alpha * rgb + (1 - alpha) * roi
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def process_frame(self, frame):
        """Process frame and detect face"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return results

    def release(self):
        """Release resources"""
        self.face_detection.close()

    def apply(self, frame, landmarks=None):
        """Aplica el modelo 3D sobre la cara detectada (rotando con la cabeza)."""
        if not landmarks:
            return frame

        # Tomar la primera cara detectada
        face = landmarks[0]

        # Obtener puntos 2D de referencia para pose (solvePnP)
        image_points = []
        for key in ('nose_tip', 'chin', 'left_eye_outer', 'right_eye_outer', 'left_mouth', 'right_mouth'):
            idx = self._face_mesh_idxs[key]
            lm = face.landmark[idx]
            x = lm.x * frame.shape[1]
            y = lm.y * frame.shape[0]
            image_points.append((x, y))

            # dibujar punto de referencia (para debugging)
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        image_points = np.array(image_points, dtype=np.float32)

        # Actualizar cámara según tamaño del frame
        h, w = frame.shape[:2]
        focal_length = w  # aproximación común: focal = ancho de la imagen en píxeles
        self.camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Resolver pose (con fallback a otro algoritmo si falla)
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
        )

        # ajustar escala de la traslación para que el modelo no quede demasiado lejos
        tvec = tvec * .15   

        # Mostrar indicador rápido de éxito/fallo (debug)
        status_text = "POSE OK" if success else "POSE FAIL"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if success else (0, 0, 255), 2)

        if not success:
            # Si no se puede estimar la pose, dibujamos una caja para comprobar landmarks
            xs = image_points[:, 0]
            ys = image_points[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            return frame

        # Dibujar ejes de la pose para debug (x=rojo, y=verde, z=azul)
        try:
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, length=50)
        except Exception:
            pass

        # Determinar escala para que el modelo cubra la cara ~ 120%
        # Usamos la distancia entre ojos en imagen como referencia.
        left_eye = image_points[2]
        right_eye = image_points[3]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        scale = eye_dist / 100.0  # 100mm es el valor aproximado usado en model_points
        # Aumentar el tamaño para que el casco sea más visible
        scale *= 10.0
        # Evitar escala 0 o muy pequeña
        scale = max(scale, 0.1)
        # Comprobar si la proyección genera puntos visibles
        frame_h, frame_w = frame.shape[:2]
        projected, _ = cv2.projectPoints(self.mesh_vertices * scale, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        projected = projected.reshape(-1, 2)
        if not ((projected[:, 0] >= 0) & (projected[:, 0] < frame_w) & (projected[:, 1] >= 0) & (projected[:, 1] < frame_h)).any():
            # Si nada queda en pantalla, dibujar la caja de referencia
            xs = image_points[:, 0]
            ys = image_points[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            return frame

        # Ajuste de offset 2D para centrar la malla en la nariz detectada.
        # Esto ayuda cuando la proyección del modelo sale del cuadro.
        nose_pos = image_points[0]
        origin_proj, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        origin_proj = origin_proj.reshape(2)
        # Desplazar hacia arriba para que el casco quede sobre la cabeza (no en el mentón)
        y_offset = int(eye_dist * 0.6)
        

        # Proyectar y dibujar el mesh
        output = self._project_mesh(frame, rvec, tvec, scale)

        # Si el casco sigue sin mostrarse, dibujamos una señal clara en la proyección del origen
        origin_proj, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        origin_xy = origin_proj.reshape(2).astype(int)
        cv2.circle(output, tuple(origin_xy), 10, (0, 0, 255), -1)

        return output
    
        print("tvec:", tvec.ravel())