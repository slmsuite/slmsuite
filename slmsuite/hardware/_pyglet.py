"""
Hidden abstract class for pyglet windowing in slmsuite.
"""
import os
import ctypes
import numpy as np

try:
    import pyglet
    import pyglet.gl as gl
    from pyglet.window import Window as __Window
except:
    __Window = object

class _Window(__Window):
    def __init__(self, shape, screen=None, caption=""):
        # Make the window and do basic setup.
        if screen is None:
            display = pyglet.canvas.get_display()
            screen = display.get_default_screen()

        if shape is None:   # Fullscreen
            super().__init__(
                screen=screen,
                fullscreen=True,
                vsync=True,
                caption=caption
            )
            self.set_mouse_visible(False)
            self.flip()
        else:
            super().__init__(
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

    def _setup_context(self):
        shape = self.shape

        if gl.base.gl_info.have_version(3,0):       # Pyglet >= 2.0.0
            B = 4

            # Setup buffers (texbuffer is power of 2 padded to init the memory in OpenGL)
            self.buffer = np.zeros(shape + (B,), dtype=np.uint8)
            N = int(shape[0] * shape[1] * B)
            self.cbuffer = (gl.GLubyte * N).from_buffer(self.buffer)

            # Setup the texture
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                shape[1], shape[0],
                0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Use the default pyglet shader; this is required in 2.0+.
            self.shader = pyglet.graphics.get_default_blit_shader()
            self.shader.use()

            # Also allocate the quadrangle using pyglet 2.0+ formalism.
            self.batch = pyglet.graphics.Batch()
            self.vertex_list = self.shader.vertex_list(
                4,
                gl.GL_TRIANGLE_STRIP,
                self.batch,
                position=('i',
                    [
                        0,  shape[0], 0,
                        0, 0, 0,
                        shape[1], shape[0], 0,
                        shape[1], 0, 0,
                    ]
                ),  # Vertex positions (x, y)
                tex_coords= ('f',
                    [
                        0., 1., 0.,
                        0., 0., 0.,
                        1., 1., 0.,
                        1., 0., 0.,
                    ]
                )   # Texture coordinates (u, v, r)
            )

            # Cleanup.
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glFlush()
        elif gl.base.gl_info.have_version(2,0):     # Pyglet < 2.0.0
            # Set the viewpoint.
            proj = pyglet.window.Projection2D()
            proj.set(shape[1], shape[0], shape[1], shape[0])

            # Setup shapes.
            texture_shape = tuple(
                np.power(2, np.ceil(np.log2(shape))).astype(np.int64)
            )
            self.tex_shape_ratio = (
                float(shape[0])/float(texture_shape[0]),
                float(shape[1])/float(texture_shape[1])
            )
            B = 4

            # Setup buffers (texbuffer is power of 2 padded to init the memory in OpenGL).
            self.buffer = np.zeros(shape + (B,), dtype=np.uint8)
            N = int(shape[0] * shape[1] * B)
            self.cbuffer = (gl.GLubyte * N).from_buffer(self.buffer)

            texbuffer = np.zeros(texture_shape + (B,), dtype=np.uint8)
            Nt = int(texture_shape[0] * texture_shape[1] * B)
            texcbuffer = (gl.GLubyte * Nt).from_buffer(texbuffer)

            # Setup the texture.
            gl.glEnable(gl.GL_TEXTURE_2D)
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_FALSE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory.
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                texture_shape[1], texture_shape[0],
                0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                texcbuffer
            )

            # Make sure we can write to a subset of the memory (as we will do in the future).
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Cleanup.
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glFlush()
        else:
            raise RuntimeError("Could not find a compatible GL context.")

    def render(self):
        self.switch_to()

        shape = self.shape

        if gl.base.gl_info.have_version(3,0):       # Pyglet >= 2.0.0
            self.shader.use()

            # Bind texture.
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Draw the quad.
            self.vertex_list.draw(gl.GL_TRIANGLE_STRIP)
        elif gl.base.gl_info.have_version(2,0):     # Pyglet < 2.0.0
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

    @staticmethod
    def info(verbose=True):
        """
        Get information about the available displays, their indexes, and their sizes.

        Parameters
        ----------
        verbose : bool
            Whether or not to print display information.

        Returns
        -------
        list of (int, (int, int, int, int), bool, bool) tuples
            The number, geometry of each display.
        """
        # Note: in pyglet, the display is the full arrangement of screens,
        # unlike the terminology in other SLM subclasses
        display = pyglet.canvas.get_display()

        screens = display.get_screens()
        default = display.get_default_screen()
        windows = display.get_windows()

        def parse_screen(screen):
            return (
                "x={}, y={}, width={}, height={}"
                .format(screen.x, screen.y, screen.width, screen.height)
            )
        def parse_screen_int(screen):
            return (screen.x, screen.y, screen.width, screen.height)
        def parse_window(window):
            x, y = window.get_location()
            return (
                "x={}, y={}, width={}, height={}"
                .format(x, y, window.width, window.height)
            )

        default_str = parse_screen(default)

        window_strs = []
        for window in windows:
            window_strs.append(parse_window(window))

        if verbose:
            print('Display Positions:')
            print('#,  Position')

        screen_list = []

        for x, screen in enumerate(screens):
            screen_str = parse_screen(screen)

            main_bool = False
            window_bool = False

            if screen_str == default_str:
                main_bool = True
                screen_str += ' (main)'
            if screen_str in window_strs:
                window_bool = True
                screen_str += ' (has ScreenMirrored)'

            if verbose:
                print('{},  {}'.format(x, screen_str))

            screen_list.append((
                x,
                parse_screen_int(screen),
                main_bool,
                window_bool
            ))

        return screen_list
