import textwrap
import matplotlib.pyplot as plt


def draw_interation(cube1='Autotune Center',
                    cube2='FS Runner',
                    arrow_dir='down',
                    arrow_len=6,
                    info=None,
                    max_width=20,
                    font_size=50):
    def wrap_text(text, text_width=20):
        if len(text) > text_width - len('...'):
            return textwrap.shorten(text, width=text_width, placeholder="...")
        else:
            return ' ' * (text_width - len(text)) + text + ' ' * (text_width -
                                                                  len(text))

    x, y = 0.3, 0.4
    size, max_len = font_size, 10
    width, height = (size / max_len) * 0.085, 0.5

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot()

    # Cubes
    ax.text(x,
            y + height,
            wrap_text(cube1, max_width),
            size=size,
            bbox=dict(boxstyle="round4", alpha=0.3, color='black'))

    ax.text(x,
            y,
            wrap_text(cube2, max_width),
            size=size,
            bbox=dict(boxstyle="round4", alpha=0.3, color='black'))

    # Arrows
    a_x = x + width / 2
    a_y = (y + height) / 2 + 0.1
    ax.text(a_x,
            a_y,
            wrap_text(' ', arrow_len),
            size=size,
            rotation=270 if arrow_dir == 'down' else 90,
            bbox=dict(boxstyle="rarrow", alpha=0.2, color='green'))

    if info:
        # Bubbles
        ix, iy = a_x + 0.08, a_y
        for i in range(3):
            ax.text(ix,
                    iy,
                    ' ',
                    size=20 + i * 3,
                    rotation=270,
                    bbox=dict(boxstyle="circle", alpha=0.4, color='lightblue'))
            ix, iy = ix + 0.03, iy + 0.03
        # Information
        info = textwrap.fill(info,
                             width=max_width,
                             max_lines=8,
                             placeholder="...")
        plt.text(ix,
                 iy,
                 info,
                 size=40,
                 bbox=dict(
                     boxstyle="sawtooth",
                     facecolor='lightblue',
                     edgecolor='black',
                 ))
    plt.axis('off')
    fig = plt.gcf()
    plt.close()
    return fig
