import importlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import trimesh


class LegoHeadFilter:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

        self.model_path = Path(__file__).parent.parent / "assets" / "lego_head.glb"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # MediaPipe indices.
        self._face_mesh_idxs = {
            "nose_tip": 1,
            "forehead": 10,
            "chin": 152,
            "left_eye_outer": 33,
            "right_eye_outer": 263,
            "left_mouth": 61,
            "right_mouth": 291,
            "left_temple": 234,
            "right_temple": 454,
        }

        # Canonical 3D head points for solvePnP (mm).
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float32,
        )

        # Nose -> neck in canonical head coordinates (mm).
        self.head_neck_offset_local = np.array([0.0, -110.0, -130.0], dtype=np.float32)

        self.focal_length = 1000.0
        self.h, self.w = 480, 640
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.w / 2], [0, self.focal_length, self.h / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        self.prev_rvec = None
        self.prev_tvec = None
        # Lower value = less lag when user rotates.
        self.pose_blend = 0.08

        # OpenCV camera coordinates -> OpenGL camera coordinates.
        self.cv_to_gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        self.fallback_reason = ""
        self.pyrender = self._load_pyrender()
        self.use_pyrender = self.pyrender is not None

        self.renderer = None
        self.render_scene = None
        self.mesh_node = None
        self.camera_node = None

        self._init_render_assets()

    def _load_pyrender(self):
        try:
            return importlib.import_module("pyrender")
        except Exception as exc:
            self.fallback_reason = f"pyrender import failed: {type(exc).__name__}: {exc}"
            return None

    def _iter_scene_trimesh(self, obj):
        if isinstance(obj, trimesh.Trimesh):
            return [(obj, np.eye(4, dtype=np.float32))]

        if isinstance(obj, trimesh.Scene):
            items = []
            graph = obj.graph
            for node_name in graph.nodes_geometry:
                transform, geom_name = graph[node_name]
                geom = obj.geometry.get(geom_name)
                if geom is None:
                    continue
                items.append((geom, np.asarray(transform, dtype=np.float32)))
            return items

        return []

    def _init_render_assets(self):
        loaded = trimesh.load(str(self.model_path))
        mesh_items = self._iter_scene_trimesh(loaded)
        if not mesh_items:
            raise RuntimeError("Empty GLB scene")

        merged = []
        for geom, pose in mesh_items:
            g = geom.copy()
            g.apply_transform(pose)
            merged.append(g)
        merged_mesh = trimesh.util.concatenate(tuple(merged))

        # Build a canonical mesh: neck anchored and normalized.
        verts = np.asarray(merged_mesh.vertices, dtype=np.float32)
        centroid = verts.mean(axis=0)
        verts = verts - centroid

        y = verts[:, 1]
        neck_cut = np.percentile(y, 10)
        neck_band = verts[y <= neck_cut]
        if len(neck_band) < 8:
            neck_anchor = np.array([0.0, float(np.min(y)), 0.0], dtype=np.float32)
        else:
            neck_anchor = neck_band.mean(axis=0)

        verts = verts - neck_anchor
        max_rad = float(np.max(np.linalg.norm(verts, axis=1)))
        if max_rad <= 0:
            max_rad = 1.0
        verts = verts / max_rad

        render_mesh = merged_mesh.copy()
        render_mesh.vertices = verts

        self.mesh_vertices = np.asarray(render_mesh.vertices, dtype=np.float32)
        self.mesh_faces = np.asarray(render_mesh.faces, dtype=np.int32)

        self.mesh_width = max(float(np.ptp(self.mesh_vertices[:, 0])), 1e-6)
        self.face_albedo_bgr = self._compute_face_albedo_fallback(render_mesh)

        if not self.use_pyrender:
            return

        try:
            self.render_scene = self.pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.28, 0.28, 0.28, 1.0])

            pm = self.pyrender.Mesh.from_trimesh(render_mesh, smooth=True)
            self.mesh_node = self.render_scene.add(pm, pose=np.eye(4, dtype=np.float32))

            cam = self.pyrender.IntrinsicsCamera(
                fx=self.focal_length,
                fy=self.focal_length,
                cx=self.w / 2,
                cy=self.h / 2,
                znear=1.0,
                zfar=5000.0,
            )
            self.camera_node = self.render_scene.add(cam, pose=np.eye(4, dtype=np.float32))

            key_light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=3.3)
            fill_light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.8)
            rim_light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)

            self.render_scene.add(key_light, pose=np.eye(4, dtype=np.float32))

            fill_pose = np.eye(4, dtype=np.float32)
            fill_pose[:3, :3] = cv2.Rodrigues(np.array([0.35, -0.95, 0.0], dtype=np.float32))[0]
            self.render_scene.add(fill_light, pose=fill_pose)

            rim_pose = np.eye(4, dtype=np.float32)
            rim_pose[:3, :3] = cv2.Rodrigues(np.array([-0.20, 2.6, 0.0], dtype=np.float32))[0]
            self.render_scene.add(rim_light, pose=rim_pose)

            self.renderer = self.pyrender.OffscreenRenderer(viewport_width=self.w, viewport_height=self.h)
        except Exception as exc:
            self.use_pyrender = False
            self.renderer = None
            self.render_scene = None
            self.mesh_node = None
            self.camera_node = None
            self.fallback_reason = f"pyrender init failed: {type(exc).__name__}: {exc}"

    def _compute_face_albedo_fallback(self, tri):
        num_faces = len(self.mesh_faces)
        fallback = np.tile(np.array([[80.0, 200.0, 245.0]], dtype=np.float32), (num_faces, 1))

        visual = getattr(tri, "visual", None)
        if visual is None:
            return fallback

        try:
            uv = getattr(visual, "uv", None)
            material = getattr(visual, "material", None)
            image = getattr(material, "image", None) if material is not None else None
            if uv is not None and image is not None:
                img = np.array(image)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                h, w = img.shape[:2]
                uvf = np.asarray(uv, dtype=np.float32)
                fuv = uvf[self.mesh_faces].mean(axis=1)
                u = np.mod(fuv[:, 0], 1.0)
                v = np.mod(fuv[:, 1], 1.0)
                px = np.clip((u * (w - 1)).round().astype(np.int32), 0, w - 1)
                py = np.clip(((1.0 - v) * (h - 1)).round().astype(np.int32), 0, h - 1)
                rgb = img[py, px, :3].astype(np.float32)
                return rgb[:, ::-1]
        except Exception:
            pass

        try:
            face_colors = getattr(visual, "face_colors", None)
            if face_colors is not None and len(face_colors) == num_faces:
                return np.asarray(face_colors)[:, :3].astype(np.float32)
        except Exception:
            pass

        return fallback

    def _solve_pose(self, image_points):
        if self.prev_rvec is not None and self.prev_tvec is not None:
            ok, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                rvec=self.prev_rvec.copy(),
                tvec=self.prev_tvec.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                return ok, rvec, tvec

        ok, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            return ok, rvec, tvec

        return cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )

    def _render_pyrender(self, frame, rvec, tvec, scale):
        if self.renderer is None or self.render_scene is None or self.mesh_node is None or self.camera_node is None:
            return frame

        h, w = frame.shape[:2]
        if w != self.w or h != self.h:
            self.w, self.h = w, h
            try:
                self.renderer.delete()
            except Exception:
                pass
            self.renderer = self.pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

            try:
                self.render_scene.remove_node(self.camera_node)
            except Exception:
                pass

            cam = self.pyrender.IntrinsicsCamera(
                fx=float(self.camera_matrix[0, 0]),
                fy=float(self.camera_matrix[1, 1]),
                cx=float(self.camera_matrix[0, 2]),
                cy=float(self.camera_matrix[1, 2]),
                znear=1.0,
                zfar=5000.0,
            )
            self.camera_node = self.render_scene.add(cam, pose=np.eye(4, dtype=np.float32))

        R, _ = cv2.Rodrigues(rvec)
        model_pose_cv = np.eye(4, dtype=np.float32)
        model_pose_cv[:3, :3] = (R * scale).astype(np.float32)
        model_pose_cv[:3, 3] = tvec.reshape(3).astype(np.float32)

        model_pose_gl = self.cv_to_gl @ model_pose_cv

        self.render_scene.set_pose(self.mesh_node, pose=model_pose_gl)
        self.render_scene.set_pose(self.camera_node, pose=np.eye(4, dtype=np.float32))

        color, _ = self.renderer.render(self.render_scene, flags=self.pyrender.RenderFlags.RGBA)
        rgb = color[:, :, :3].astype(np.float32)
        alpha = (color[:, :, 3:4].astype(np.float32)) / 255.0

        out = frame.astype(np.float32)
        out = alpha * rgb[:, :, ::-1] + (1.0 - alpha) * out
        return out.astype(np.uint8)

    def _render_fallback_cpu(self, frame, rvec, tvec, scale):
        verts_local = self.mesh_vertices * scale
        R, _ = cv2.Rodrigues(rvec)
        verts_cam = verts_local @ R.T + tvec.reshape(1, 3)

        projected, _ = cv2.projectPoints(
            verts_cam,
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            self.camera_matrix,
            self.dist_coeffs,
        )
        pts2d = projected.reshape(-1, 2)

        faces = self.mesh_faces
        v0 = verts_cam[faces[:, 0]]
        v1 = verts_cam[faces[:, 1]]
        v2 = verts_cam[faces[:, 2]]

        valid = (v0[:, 2] > 1e-3) & (v1[:, 2] > 1e-3) & (v2[:, 2] > 1e-3)
        if not np.any(valid):
            return frame

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-6)

        intensity = 0.54 + 0.66 * np.clip(-normals[:, 2], 0.0, 1.0)
        face_colors = np.clip(self.face_albedo_bgr * intensity[:, None], 0.0, 255.0).astype(np.uint8)

        depth = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
        draw_ids = np.where(valid)[0]
        draw_ids = draw_ids[np.argsort(depth[draw_ids])[::-1]]

        overlay = np.zeros_like(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for i in draw_ids:
            tri = np.round(pts2d[faces[i]]).astype(np.int32)
            cv2.fillConvexPoly(
                overlay,
                tri,
                color=(int(face_colors[i, 0]), int(face_colors[i, 1]), int(face_colors[i, 2])),
                lineType=cv2.LINE_AA,
            )
            cv2.fillConvexPoly(mask, tri, 255, lineType=cv2.LINE_AA)

        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        out = frame.astype(np.float32)
        out = alpha * overlay.astype(np.float32) + (1.0 - alpha) * out

        out_u8 = out.astype(np.uint8)
        cv2.putText(out_u8, "CPU FALLBACK", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        if self.fallback_reason:
            msg = self.fallback_reason[:88]
            cv2.putText(out_u8, msg, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        return out_u8

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_detection.process(rgb_frame)

    def release(self):
        self.face_detection.close()
        if self.renderer is not None:
            try:
                self.renderer.delete()
            except Exception:
                pass

    def apply(self, frame, landmarks=None):
        if not landmarks:
            return frame

        face = landmarks[0]

        image_points = []
        for key in ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth"):
            idx = self._face_mesh_idxs[key]
            lm = face.landmark[idx]
            image_points.append((lm.x * frame.shape[1], lm.y * frame.shape[0]))
        image_points = np.array(image_points, dtype=np.float32)

        def lm_xy(key):
            lm = face.landmark[self._face_mesh_idxs[key]]
            return np.array([lm.x * frame.shape[1], lm.y * frame.shape[0]], dtype=np.float32)

        forehead_xy = lm_xy("forehead")
        chin_xy = lm_xy("chin")
        left_eye_xy = lm_xy("left_eye_outer")
        right_eye_xy = lm_xy("right_eye_outer")
        left_temple_xy = lm_xy("left_temple")
        right_temple_xy = lm_xy("right_temple")

        h, w = frame.shape[:2]
        focal_length = float(w)
        self.camera_matrix = np.array(
            [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )

        success, rvec, tvec = self._solve_pose(image_points)
        if not success:
            return frame

        if self.prev_rvec is None:
            self.prev_rvec = rvec.copy()
            self.prev_tvec = tvec.copy()
        else:
            b = self.pose_blend
            self.prev_rvec = b * self.prev_rvec + (1.0 - b) * rvec
            self.prev_tvec = b * self.prev_tvec + (1.0 - b) * tvec

        rvec = self.prev_rvec
        tvec = self.prev_tvec

        z = float(abs(tvec[2, 0])) if tvec.ndim == 2 else float(abs(tvec[2]))
        z = max(z, 1e-3)

        temple_px = float(np.linalg.norm(left_temple_xy - right_temple_xy))
        eye_px = float(np.linalg.norm(left_eye_xy - right_eye_xy))
        face_height_px = float(np.linalg.norm(chin_xy - forehead_xy))

        # Helmet sizing: bigger than head but still locked to neck.
        target_head_px = max(temple_px * 1.42, eye_px * 3.15, face_height_px * 1.68, 90.0)
        scale = (target_head_px * z / focal_length) / max(self.mesh_width, 1e-6)
        scale = max(scale, 1e-3)

        # Move from nose anchor (solvePnP) to real neck anchor in camera coordinates.
        R, _ = cv2.Rodrigues(rvec)
        neck_offset_cam = (R @ self.head_neck_offset_local.reshape(3, 1)).reshape(3, 1)

        # Slight extra lift so helmet shell sits above scalp while neck stays aligned.
        up_lift_local = np.array([0.0, 18.0, 0.0], dtype=np.float32).reshape(3, 1)
        tvec = tvec + neck_offset_cam + (R @ up_lift_local)

        if self.use_pyrender:
            return self._render_pyrender(frame, rvec, tvec, scale)

        return self._render_fallback_cpu(frame, rvec, tvec, scale)
