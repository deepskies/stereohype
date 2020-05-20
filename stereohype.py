import numpy as np
import matplotlib.pyplot as plt
import skimage, skimage.io
import time


def blank_image(shape=(600, 800, 4), rgba=(255, 255, 255, 0)):
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
    plt.show()


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
    return np.random.randint(0, levels, shape) / levels


def create_circular_depthmap(shape, center=None, radius=None):
    "Creates a circular depthmap, centered on the image."
    n, h, w = shape

    r = np.arange(h)
    c = np.arange(w)
    r, c = np.meshgrid(r, c, indexing='ij')
    r = np.repeat(r[:, :, np.newaxis], n, axis=2)
    c = np.repeat(c[:, :, np.newaxis], n, axis=2)
    if center is None:
        center = np.array([(h-1) / 2, (w-1) / 2])*np.ones((n, 2))
    if radius is None:
        radius = 100*np.ones(n)
    d = np.sqrt((r - center[:, 0])**2 + (c - center[:, 1])**2)
    depthmap = (d < radius).astype(float)
    depthmap = np.transpose(depthmap, (2, 0, 1))
    return depthmap


def normalize(depthmap):
    "Normalizes values of depthmap to [0, 1] range."
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap


def make_autostereogram(shape, depthmap, pattern, shift_amplitude=0.1, invert=False):
    "Creates an autostereogram from depthmap and pattern."
    if invert:
        depthmap = 1 - depthmap
    n, h, w = shape
    autostereogram = np.zeros(shape, dtype=pattern.dtype)
    w_pattern = pattern.shape[2]
    shift = (depthmap * shift_amplitude * w_pattern).astype(int)
    autostereogram[:, :, :w_pattern] = pattern
    grid = np.indices((n, h))
    c = np.arange(w)
    pos = c - w_pattern + shift

    for c in np.arange(w_pattern, w):
        autostereogram[:, :, c] = autostereogram[grid[0], grid[1], pos[:, :, c]]
    return autostereogram


def generate_data(n_obj=1, radius_random=True, verbose=False, invert=False, save=False):

    # hyper params
    version = 000

    # shape parameters
    width_image = 512
    height_image = width_image
    shape_image = (n_obj, height_image, width_image)
    shape_depth = shape_image
    width_pattern = 64
    height_pattern = height_image
    shape_pattern = (n_obj, height_pattern, width_pattern)
    radius_min = 50
    radius_max = 200
    center_depth = [256, 270]
    center_depth = np.array(center_depth) * np.ones((n_obj, 2))

    version_str = str(version)
    width_image_str = str(width_image)
    width_pattern_str = str(width_pattern)

    file_image = "data_v" + version_str + "_circle_radius_" + width_image_str + "_" + width_pattern_str + "_image.npy"
    file_depth = "data_v" + version_str + "_circle_radius_" + width_image_str + "_" + width_pattern_str + "_depth.npy"
    print(file_image)
    print(file_depth)

    # depth maps
    print("Generating depth map-based autostereograms")
    image = np.zeros((n_obj, width_image, height_image, 1))

    t0 = time.time()

    # Generate Pattern
    pattern = make_pattern(shape=shape_pattern)
    if verbose:
        display(pattern[0])

    # select radius
    if radius_random:
        radius = np.random.uniform(radius_min, radius_max, n_obj)
    else:
        radius = 100*np.ones(n_obj)

    # create depth map
    depth = create_circular_depthmap(shape_depth, center=center_depth, radius=radius)

    if verbose:
        display(depth[0], colorbar=True)

    # normalize
    depth = normalize(depth)

    image = make_autostereogram(shape_image, depth, pattern, invert=invert)

    if verbose:
        display(image[0])

    depth = np.reshape(depth, shape_depth + (1,))
    image = np.reshape(image, shape_image + (1,))

    if save:
        np.save(file_image, image)
        np.save(file_depth, depth)

    t1 = time.time()

    dt = t1-t0
    tavg = dt/float(n_obj)
    print("average time per object", tavg)

    return image, depth


def test():

    print("running out of the box test")

    plt.rcParams['figure.dpi'] = 150
    file_tile ="coin-icon-2.png"

    # patterns
    img = blank_image()
    #display(img)
    coin = skimage.io.imread(file_tile)
    print(coin.shape)
    print(img.shape)
    print("read coin image file")
    #display(coin)
    test_img = insert_pattern(img, coin, (10, 20))
    print("make test image")
    display(test_img)
    test_img = tile_horizontally(img, coin, (10, 20), 3, 128)
    print("make multiple test image")
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


def main():
    generate_data(n_obj=1, radius_random=True, verbose=False, invert=False, save=True)


if __name__ == "__main__":
    main()
