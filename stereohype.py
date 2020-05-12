import numpy as np
import matplotlib.pyplot as plt
import skimage, skimage.io


def blank_image(shape=(600, 800, 4),
               rgba=(255, 255, 255, 0)):
    "Returns a blank image, of size defined by shape and background color rgb."
    return np.ones(shape, dtype=np.float) * np.array(rgba) / 255.



def display(img, colorbar=False):
    "Displays an image."
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 2:
        i = skimage.io.imshow(img, cmap='gray')
    else:
        i = skimage.io.imshow(img)
    if colorbar:
        plt.colorbar(i, shrink=0.5, label='depth')
    plt.tight_layout()



def insert_pattern(background_img, pattern, location):
    """Inserts a pattern onto a background, at given location. Returns new image."""
    img = background_img.copy()
    r0, c0 = location
    r1, c1 = r0 + pattern.shape[0], c0 + pattern.shape[1]
    if r1 < background_img.shape[0] and c1 < background_img.shape[1]:
        img[r0:r1, c0:c1, :] = skimage.img_as_float(pattern)
    return img


def tile_horizontally(background_img, pattern, start_location, repetitions, shift):
    "Tiles a pattern on a background image, repeatedly with a given shift."
    img = background_img.copy()
    for i in range(repetitions):
        r, c = start_location
        c += i * shift
        img = insert_pattern(img, pattern, location=(r, c))
    return img


def make_pattern(shape=(16, 16), levels=64):
    "Creates a pattern from gray values."
    return np.random.randint(0, levels - 1, shape) / levels


def create_circular_depthmap(shape=(600, 800), center=None, radius=100):
    "Creates a circular depthmap, centered on the image."
    depthmap = np.zeros(shape, dtype=np.float)
    r = np.arange(depthmap.shape[0])
    c = np.arange(depthmap.shape[1])
    R, C = np.meshgrid(r, c, indexing='ij')
    if center is None:
        center = np.array([r.max() / 2, c.max() / 2])
    d = np.sqrt((R - center[0])**2 + (C - center[1])**2)
    depthmap += (d < radius)
    return depthmap



def normalize(depthmap):
    "Normalizes values of depthmap to [0, 1] range."
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap




def make_autostereogram(depthmap, pattern, shift_amplitude=0.1, invert=False):
    "Creates an autostereogram from depthmap and pattern."
    depthmap = normalize(depthmap)
    if invert:
        depthmap = 1 - depthmap
    autostereogram = np.zeros_like(depthmap, dtype=pattern.dtype)
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            else:
                shift = int(depthmap[r, c] * shift_amplitude * pattern.shape[1])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
    return autostereogram






def main():

    plt.rcParams['figure.dpi'] = 150

    # patterns
    img = blank_image()
    display(img)
    coin = skimage.io.imread('files/coin-icon.png')
    display(coin)
    test_img = insert_pattern(img, coin, (10, 20))
    display(test_img)
    test_img = tile_horizontally(img, coin, (10, 20), 3, 128)
    display(test_img)

    img = blank_image(shape=(450, 800, 4))
    img = tile_horizontally(img, coin, (10, 10), 6, shift=130)
    img = tile_horizontally(img, coin, (10 + 150, 10), 5, shift=150)
    img = tile_horizontally(img, coin, (10 + 2*150, 10), 5, shift=140)
    display(img)

    # depth maps
    pattern = make_pattern(shape=(128, 64))
    display(pattern)
    depthmap = create_circular_depthmap(radius=150)
    display(depthmap, colorbar=True)
    autostereogram = make_autostereogram(depthmap, pattern)
    display(autostereogram)


    autostereogram = make_autostereogram(depthmap, pattern, invert=True)
    display(autostereogram)



    depthmap = create_circular_depthmap(center=(200, 300), radius=100) + \
           create_circular_depthmap(center=(450, 500), radius=100) + \
           create_circular_depthmap(center=(200, 550), radius=150)
    depthmap = normalize(depthmap)
    display(depthmap, colorbar=True)
    autostereogram = make_autostereogram(depthmap, pattern)
    display(autostereogram)
