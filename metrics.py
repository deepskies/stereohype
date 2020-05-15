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

