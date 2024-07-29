"""
Hidden abstract class for windowing in slmsuite.
"""
import pyglet
import pyglet.gl as gl
import os
import ctypes
import numpy as np
# import pyglet
# from pyglet.gl import *
from pyglet.math import Mat4


class _Window(pyglet.window.Window):
    def __init__ (self, shape, screen=None, caption=""):
        # Make the window and do basic setup.
        if screen is None:
            display = pyglet.canvas.get_display()
            screen = display.get_default_screen()

        self.alive = True

        if shape is None:   # Fullscreen
            super(_Window, self).__init__(
                screen=screen,
                fullscreen=True,
                vsync=True,
                caption=caption
            )
            self.set_mouse_visible(False)
        else:
            super(_Window, self).__init__(
                screen=screen,
                width=shape[1],
                height=shape[0],
                resizable=True,
                fullscreen=False,
                vsync=True,
                caption=caption,
                style=pyglet.window.Window.WINDOW_STYLE_DEFAULT
            )
            self.set_visible(False)
            self.flip()

        self.shape = (self.height, self.width)

        try:
            # Icons. Currently hardcoded. Feel free to implement custom icons.
            path, _ = os.path.split(os.path.realpath(__file__))
            path = os.path.join(
                path, '..', '..', 'docs', 'source', 'static', 'slmsuite-notext-'
            )
            img16x16 =      pyglet.image.load(path + '16x16.png')
            img32x32 =      pyglet.image.load(path + '32x32.png')
            img512x512 =    pyglet.image.load(path + '512x512.png')
            self.set_icon(img16x16, img32x32, img512x512)
        except Exception as e:
            print(e)

        self._setup_context()

    def _setup_context(self):
        shape = self.shape

        if gl.base.gl_info.have_version(3,0):
                        # Create an image or load an existing image
            image = pyglet.image.SolidColorImagePattern(
                (255, 255, 255, 255)
            ).create_image(256, 256)
            texture = image.get_texture()

            # Flush image upload before data get GC'd:
            gl.glFlush()
        elif gl.base.gl_info.have_version(2,0):
            # Set the viewpoint.
            proj = pyglet.window.Projection2D()
            proj.set(shape[1], shape[0], shape[1], shape[0])

            # Setup shapes
            texture_shape = tuple(
                np.power(2, np.ceil(np.log2(shape))).astype(np.int64)
            )
            self.tex_shape_ratio = (
                float(shape[0])/float(texture_shape[0]),
                float(shape[1])/float(texture_shape[1])
            )
            B = 4

            # Setup buffers (texbuffer is power of 2 padded to init the memory in OpenGL)
            self.buffer = np.zeros(shape + (B,), dtype=np.uint8)
            N = int(shape[0] * shape[1] * B)
            self.cbuffer = (gl.GLubyte * N).from_buffer(self.buffer)

            texbuffer = np.zeros(texture_shape + (B,), dtype=np.uint8)
            Nt = int(texture_shape[0] * texture_shape[1] * B)
            texcbuffer = (gl.GLubyte * Nt).from_buffer(texbuffer)

            # Setup the texture
            gl.glEnable(gl.GL_TEXTURE_2D)
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_FALSE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                texture_shape[1], texture_shape[0],
                0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                texcbuffer
            )

            # Make sure we can write to a subset of the memory (as we will do in the future)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Cleanup
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glFlush()
        else:
            raise RuntimeError("Could not find a compatible GL context.")


    def render(self):
        self.switch_to()

        shape = self.shape

        if gl.base.gl_info.have_version(3,0):
                        # Create an image or load an existing image
            image = pyglet.image.SolidColorImagePattern(
                (255, 255, 255, 255)
            ).create_image(256, 256)
            texture = image.get_texture()

            # self.projection = Mat4.orthogonal_projection(
            #     0, shape[1], 0, shape[0], -255, 255
            # )
            # vertex_list = program.vertex_list_indexed(
            # 4, GL_TRIANGLES, [0, 1, 2, 0, 2, 3], batch, group,
            # colors=('Bn', (255, 255, 255, 255) * 4),
            # tex_coords=('f', texture.tex_coords))
        elif gl.base.gl_info.have_version(2,0):
            # Setup texture variables.
            x1 = 0
            y1 = 0
            x2 = shape[1]
            y2 = shape[0]

            xa = 0
            ya = 0
            xb = self.tex_shape_ratio[1]
            yb = self.tex_shape_ratio[0]

            array = (gl.GLfloat * 32)(
                xa, ya, 0., 1.,         # tex coord,
                x1, y1, 0., 1.,         # real coord, ...
                xb, ya, 0., 1.,
                x2, y1, 0., 1.,
                xb, yb, 0., 1.,
                x2, y2, 0., 1.,
                xa, yb, 0., 1.,
                x1, y2, 0., 1.
            )

            # Update the texture with the cbuffer.
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Blit the texture.
            gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
            gl.glInterleavedArrays(gl.GL_T4F_V4F, 0, array)
            gl.glDrawArrays(gl.GL_QUADS, 0, 4)
            gl.glPopClientAttrib()

        # Display the other side of the double buffer.
        # (with vsync enabled, this will block until the next frame is ready to display).
        self.flip()

class _CameraWindow(_Window):
    def disable(self):
        self.set_visible(False)
        self.flip()
        self.alive = False

    def enable(self):
        self.set_visible(True)
        self.flip()
        self.alive = True

    def on_activate(self):
        print("on_activate")

    def on_draw(self):
        self.render()

    def on_close(self):
        print("on_close")
        self.disable()

    def on_hide(self):
        print("on_hide")

    def on_mouse_press(self, x, y, button, modifier):
        print("on_click")
        print(x, y, button, modifier)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        print("on_mouse_scroll")

    def on_mouse_drag(self, x, y, dx, dy, button, modifier):
        print("on_mouse_drag")

    def on_key_press(self, symbol, modifiers):
        print("on_key_press")
        if symbol == pyglet.window.key.ESCAPE: # [ESC]
            self.disable()
