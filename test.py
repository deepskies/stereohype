import stereohype as sh
import metrics

def test_metrics():
    image, depth = sh.generate_data(Nobj=1, radius_random=True, verbose=False, invert=False, save=False)
    metrics.plot_map_difference(image[0], image[0], fft=False, residual=False, percentage=False, save=False)


test_metrics()
