import mediapipe as mp
import cv2 as cv

class FaceDetector:
    def __init__(self, is_static: bool, max_num_faces: int, use_landmarks: bool, 
                 min_confidence_detection: float, min_confidence_tracking: float):
        """
        Inicializa el detector de caras.
        
        Args:
            is_static: True para imágenes estáticas, False para video
            max_num_faces: Número máximo de caras a detectar
            use_landmarks: Si se deben detectar los puntos de referencia (landmarks)
            min_confidence_detection: Confianza mínima para la detección
            min_confidence_tracking: Confianza mínima para el seguimiento
        """
        self.is_static = is_static
        self.max_num_faces = max_num_faces
        self.use_landmarks = use_landmarks
        self.min_confidence_detection = min_confidence_detection
        self.min_confidence_tracking = min_confidence_tracking        
        # Módulo de FaceMesh
        self.face_mesh_module = mp.solutions.face_mesh
        
        # Crear detector FaceMesh
        self.face_mesh = self.face_mesh_module.FaceMesh(
            static_image_mode=self.is_static,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.use_landmarks,
            min_detection_confidence=self.min_confidence_detection,
            min_tracking_confidence=self.min_confidence_tracking
        )
        
        # Módulo para dibujar
        self.drawing_utils = mp.solutions.drawing_utils
    
    def detect(self, image):

        # Convertir BGR -> RGB
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Procesar la imagen con FaceMesh
        results = self.face_mesh.process(rgb_image)
        
        # Retornar el listado de mallas
        return results.multi_face_landmarks
    
    def draw(self, image):
        # Obtener los landmarks usando detect
        face_landmarks = self.detect(image)
        
        # Hacer una copia de la imagen para dibujar sobre ella
        output_image = image.copy()
        
        # Dibujar los landmarks si hay detectados
        #if face_landmarks:
        #    for landmarks in face_landmarks:
         #       self.drawing_utils.draw_landmarks(
          #          image=output_image,
           #         landmark_list=landmarks,
            #        connections=self.face_mesh_module.FACEMESH_TESSELATION,
             #       landmark_drawing_spec=None,
              #     connection_drawing_spec=self.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
               # )
        
        # Retornar la imagen con la teselación dibujada
        return output_image
    
    def landmark_xy(self, image, landmarks):
        # Obtener dimensiones de la imagen
        height, width = image.shape[:2]
        
        # Convertir coordinates normalizadas a píxeles
        xy_coordinates = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            xy_coordinates.append((x, y))
        
        return xy_coordinates