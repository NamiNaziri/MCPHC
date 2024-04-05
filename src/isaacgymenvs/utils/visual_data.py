from dataclasses import dataclass

import numpy as np
from vispy import scene
from vispy.scene import ViewBox
from vispy.visuals import PlaneVisual
from vispy.visuals.transforms import STTransform, MatrixTransform


class BeatSaberVisualDataContainer:

    def __init__(self, view: ViewBox):
        self.view = view
        plane = scene.visuals.Plane(
            width=3,
            height=3,
            color=(0.5, 0.5, 0.5, 1),
            parent=view.scene
        )
        self.body_spheres = []
        body_colors = [(1, 0, 0, 0.2), (0, 1, 0, 0.2), (0, 0, 1, 0.2)]
        self.shadow_ellipses = []
        self.saber_lines = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            parent=view.scene,
        )
        self.shadow_lines = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            parent=plane,
        )
        self.shadow_lines.set_gl_state(depth_test=False)
        for i in range(3):
            body_sphere = scene.visuals.Sphere(
                color=body_colors[i],
                edge_color=(0, 0, 0, 0),
                parent=view.scene,
                radius=0.1,
            )
            body_sphere.transform = STTransform(scale=(1.0, 1.0, 1.0), translate=(0, 0, 0))
            self.body_spheres.append(body_sphere)

            shadow_ellipse = scene.visuals.Ellipse(
                center=np.array([0, 0, 0]),
                radius=(0.1, 0.1),
                color=(0, 0, 0, 0.5),
                parent=plane,
            )
            self.shadow_ellipses.append(shadow_ellipse)
            shadow_ellipse.set_gl_state(depth_test=False)

        # Render order
        for body_sphere in self.body_spheres:
            body_sphere.order = 0
        plane.order = 1


class VisualDataContainer:

    def __init__(self, view: ViewBox):
        self.view = view
        plane = scene.visuals.Plane(
            width=6,
            height=6,
            color=(0.5, 0.5, 0.5, 1),
            parent=view.scene
        )
        # shadow_image = scene.visuals.Image(
        #     parent=plane,
        # )
        # shadow_canvas = scene.canvas.SceneCanvas(show=False, )
        # shadow_view = scene.widgets.ViewBox()
        self.body_markers = []
        self.body_lines = []
        self.sparse_markers = scene.visuals.Markers(
            pos=np.array([[0, 0, 0]]),
            parent=view.scene,
            size=0.1,
            edge_width=0.01,
            scaling=True,
        )
        # self.axis = scene.visuals.XYZAxis(parent=view.scene)
        self.sparse_axes = []
        for i in range(3):
            axis = scene.visuals.XYZAxis(
                # width=(i + 1) * 2,
                width=1,
                parent=view.scene
            )
            axis.transform = MatrixTransform(matrix=np.eye(4))
            self.sparse_axes.append(axis)

        self.shadow_ellipses = []
        self.shadow_markers = []
        for i in range(3):
            body_markers = scene.visuals.Markers(
                pos=np.array([[0, 0, 0]]),
                parent=view.scene,
                size=0.1,
                edge_width=0.01,
                scaling="visual",
                # spherical=True,
            )
            # body_markers.set_gl_state(depth_test=False)
            self.body_markers.append(body_markers)
            body_lines = scene.visuals.Line(parent=view.scene)
            self.body_lines.append(body_lines)

            shadow_markers = scene.visuals.Markers(
                parent=view.scene,
                size=0.1,
                edge_width=0.01,
                scaling=True,
            )
            # shadow_markers.set_gl_state(depth_test=False)
            self.shadow_markers.append(shadow_markers)

        # for i in range(22):
        #     shadow_ellipse = scene.visuals.Ellipse(
        #         center=np.array([0, 0, 0]),
        #         radius=(0.1, 0.1),
        #         color=(0, 0, 0, 0.5),
        #         parent=plane,
        #         num_segments=12,
        #     )
        #     self.shadow_ellipses.append(shadow_ellipse)
        #     shadow_ellipse.set_gl_state(depth_test=False)

        # Render order
        i = 0
        # for ax in self.sparse_axes:
        #     ax.order = i
        # i += 1
        self.body_markers[2].order = i
        i += 1
        self.body_markers[1].order = i
        i += 1
        self.body_markers[0].order = i
        i += 1
        plane.order = i
        # i += 1
        # self.shadow_markers[2].order = i
        # # i += 1
        # self.shadow_markers[1].order = i
        # i += 1
        # self.shadow_markers[0].order = i
        # for shadow_ellipse in self.shadow_ellipses:
        #     shadow_ellipse.order = 1
