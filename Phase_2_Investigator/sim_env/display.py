"""A plain image window with keyboard input, using GLFW (the same window library
as the MuJoCo viewer), for the sensor viewer and the simple teleop window.

Why not cv2.imshow: the opencv-python wheel draws windows with its own bundled
Qt, which needs X11 libraries (libxcb-*) that a fresh WSL Ubuntu does not have
("Could not load the Qt platform plugin xcb"). GLFW only needs what the MuJoCo
viewer already uses, so if the viewer opens, this opens too.

    win = ImageWindow("title", 1280, 720)
    while not win.closed:
        win.show(rgb_uint8)          # any size; scaled to fit, aspect kept
        for key in win.keys():       # GLFW key codes (W=87, Space=32, Esc=256)
            ...
    win.close()
"""
from __future__ import annotations

import queue

import numpy as np


class ImageWindow:
    def __init__(self, title, width=1280, height=720):
        import glfw
        import mujoco

        self._glfw, self._mj = glfw, mujoco
        if not glfw.init():
            raise RuntimeError("GLFW could not initialise (no display? check WSLg / $DISPLAY)")
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            raise RuntimeError("GLFW could not create a window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        # mjr_drawPixels needs a MuJoCo render context; any model will do.
        self._model = mujoco.MjModel.from_xml_string("<mujoco/>")
        self._ctx = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._ctx)
        self._keys = queue.SimpleQueue()

        def on_key(_win, key, _scancode, action, _mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                self._keys.put(key)

        glfw.set_key_callback(self.window, on_key)

    @property
    def closed(self):
        return self.window is None or self._glfw.window_should_close(self.window)

    def keys(self):
        """Keys pressed since the last call (also processes window events)."""
        self._glfw.poll_events()
        out = []
        while not self._keys.empty():
            out.append(self._keys.get_nowait())
        return out

    def show(self, rgb):
        if self.closed:
            return
        import cv2   # image resizing only (no cv2 GUI)

        glfw, mj = self._glfw, self._mj
        glfw.make_context_current(self.window)     # offscreen renderers may have switched it
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self._ctx)
        W, H = glfw.get_framebuffer_size(self.window)
        if W <= 0 or H <= 0:
            return
        h, w = rgb.shape[:2]
        s = min(W / w, H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        canvas = np.zeros((H, W, 3), np.uint8)
        y0, x0 = (H - nh) // 2, (W - nw) // 2
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(np.ascontiguousarray(rgb[..., :3]), (nw, nh),
                                                    interpolation=interp)
        mj.mjr_drawPixels(np.ascontiguousarray(canvas[::-1]).reshape(-1), None,
                          mj.MjrRect(0, 0, W, H), self._ctx)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window is not None:
            self._ctx.free()
            self._glfw.destroy_window(self.window)
            self.window = None
