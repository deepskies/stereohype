import numpy as np
import matplotlib.pyplot as plt
import time



# plot distribution of mse's for true/pred

def plot_power_spectrum(map_predict, map_true, residual=False, save=False):
    '''
    plot the power spectra and the difference

    inputs
        map, predicted
        map, true

    outputs
        figure

    help from: https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    '''

    # frequency values
    kfreq = np.fft.fftfreq(map_true.shape[0]) * map_true.shape[0]
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()

    # calculate 2d fft
    map_true_fourier = np.fft.fft2(map_true)
    map_predict_fourier = np.fft.fft2(map_predict)

    # amplitudes
    fourier_amplitudes_true = np.abs(map_true_fourier)**2
    fourier_amplitudes_true = fourier_amplitudes_true.flatten()

    fourier_amplitudes_predicted = np.abs(map_predicted_fourier)**2
    fourier_amplitudes_predicted = fourier_amplitudes_predicted.flatten()

    kbins = np.arange(0.5, map_true.shape[0]/2. + 1., 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes_true,
                                         statistic = "mean",
                                         bins = kbins)
    Abins_true *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)


    Abins_predicted, _, _ = stats.binned_statistic(knrm, fourier_amplitudes_predicted,
                                         statistic = "mean",
                                         bins = kbins)
    Abins_predicted *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)

    plt.loglog(kvals, Abins_true)
    plt.loglog(kvals, Abins_predicted)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.tight_layout()

    if save:
        plt.savefig("power_spectrum.png", dpi = 300, bbox_inches = "tight")




def plot_map_difference(map_predict, map_true, fft=False, residual=False, percentage=False, save=False):
    '''
    plot a map that is the difference between two maps

    inputs
        map, predicted
        map, true

    outpus
        figure

    '''

    # calculate 2d fft
    if fft:
        map_true = np.fft.fft2(map_true)
        map_predict = np.fft.fft2(map_predict)

    # calculate difference
    map_difference = map_true - map_predict

    # calculate residual
    if residual:
        map_difference /= map_true
        if percentage:
            map_difference *= 100.

    # plot images
    fig = plt.figure(figsize=(3*map_true.shape[0] + 30, map_true.shape[1]))

    fig.add_suplot(0, 0, 0)
    plt.imshow(map_true)

    fig.add_subplot(0, 1, 0)
    plt.imshow(map_predicted)

    fig.add_subplot(0, 2, 0)
    plt.imshow(map_difference)

    if colorbar:
        plt.colorbar(map_difference)


    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(file_fig)




def plot_history(epochs, metric_train, metric_name, metric_valid=None, save=False, figsize=(10,10)):
    '''
    plot loss history for training and validation

    inputs
        epochs
        metric --- loss or accuracy
        metric name --- "loss" or "accuracy"

    outpus
        figure

    '''

    plt.figure(figsize=figsize)
    plt.plot(epochs, metric_train)
    if metric_valid is not None:
        plt.plot(epochs, metric_valid)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.show()
    if save:
        plt.savefig(file_fig)


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


def make_pattern(shape=(16, 16), levels=64, seed=0):
    "Creates a pattern from gray values."
    np.random.seed(seed)
    return np.random.randint(0, levels - 1, shape) / levels


def create_circular_depthmap(shape, center=None, radius=100):
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




def make_autostereogram(shape, depthmap, pattern, shift_amplitude=0.1, invert=False):
    "Creates an autostereogram from depthmap and pattern."
    depthmap = normalize(depthmap)
    if invert:
        depthmap = 1 - depthmap
    autostereogram = np.zeros(shape, dtype=pattern.dtype)
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            else:
                shift = int(depthmap[r, c] * shift_amplitude * pattern.shape[1])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
    return autostereogram



def generate_data(Nobj=1, radius_random=True, verbose=False, invert=False, save=False):

    # hyper params
    version = 000

    # shape parameters
    width_image = 512
    height_image = width_image
    shape_image = (width_image, height_image)
    shape_depth = shape_image
    width_pattern = 64
    height_pattern = 512
    radius_min = 50
    radius_max = 200
    center_depth = [256, 270]

    version_str = str(version)
    width_image_str = str(width_image)
    width_pattern_str = str(width_pattern)

    file_image = "data_v" + version_str + "_circle_radius_" + width_image_str + "_" + width_pattern_str + "_image.npy"
    file_depth = "data_v" + version_str + "_circle_radius_" + width_image_str + "_" + width_pattern_str + "_depth.npy"
    print(file_image)
    print(file_depth)

    # Generate Pattern
    pattern = make_pattern(shape=(height_pattern, width_pattern))
    if verbose:
        display(pattern)

    # depth maps
    print("Generating depth map-based autostereograms")
    image = np.zeros((width_image, height_image, Nobj), pattern.dtype)
    depth = np.zeros((width_image, height_image, Nobj), pattern.dtype)


    t0 = time.time()
    for iobj in range(Nobj):

        # select radius
        if radius_random:
            radius = np.random.uniform(radius_min, radius_max)
        else:
            radius = 100

        # create depth map
        depthmap = create_circular_depthmap(shape_depth, center=center_depth, radius=radius)

        if verbose:
            display(depthmap, colorbar=True)

        # normalize
        depthmap = normalize(depthmap)

        # generate stereogram
        autostereogram = make_autostereogram(shape_image, depthmap, pattern, invert=invert)

        if verbose:
            display(autostereogram)

        # add to array fo rsaving later.
        image[:, :, iobj] = autostereogram
        depth[:, :, iobj] = depthmap

    if save:
        np.save(file_image, image)
        np.save(file_depth, depth)

    t1 = time.time()

    display(image[:,:,0])

    dt = t1-t0
    tavg = dt/float(Nobj)
    print("average time per object", tavg)





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
    generate_data(Nobj=1, radius_random=True, verbose=False, invert=False, save=False)


if __name__ == "__main__":
    main()
