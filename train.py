from utils.networks import UNet
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkc = tfk.callbacks


def main():
    batch_size = 8
    model_dir = 'unet/'
    in_data = np.load('data_v0_circle_radius_512_64_image.npy')
    out_data = np.load('data_v0_circle_radius_512_64_image.npy')
    n = len(in_data)
    n_train = int(0.9*n)
    n_val = int(0.1*n)
    in_train = in_data[:n_train]
    out_train = out_data[:n_train]
    in_val = in_data[n_train:]
    out_val = out_data[n_train:]

    nn = UNet(1, 1, features=[32, [64, 2], 64, [128, 2], 128, [256, 2], 256], multitasklength=0)

    nn.nnet.compile(loss='mean_squared_error', optimizer=tfk.optimizers.Adam(lr=0.01))
    nn.nnet.summary()

    nn.nnet.fit(in_train, out_train, epochs=100, batch_size=batch_size,
                validation_data=(in_val, out_val), verbose=True,
                callbacks=[tfkc.ModelCheckpoint(model_dir + 'best_model.h5', save_best_only=True),
                           tfkc.EarlyStopping(patience=20, verbose=1),
                           tfkc.ReduceLROnPlateau(patience=3, verbose=1, factor=0.5),
                           tfkc.CSVLogger(model_dir + 'log.csv')])


if __name__ == "__main__":
    main()