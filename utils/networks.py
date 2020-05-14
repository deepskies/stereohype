from __future__ import print_function
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class UNet(object):
    def __init__(self, inchannel, outchannel, npix, actfn="relu", features=None, bn=True, dropout=0.,
                 initialiser='glorot_uniform', l1reg=0, bias=True, skipconn=True, filtersize=3, initfiltersize=None,
                 multitasklength=1, resnets=True, max_pool=False):

        self.npix = npix
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.actfn = actfn

        if features is None:
            features = [8, 8, 16]
        if initfiltersize is None:
            initfiltersize = [3, 0]

        self.num_layers = len(features)
        nl = self.num_layers

        print('Features: ', features)
        print('Multitask length: ', multitasklength)
        print('Activation function: ', actfn)
        print('Dropout level: ', dropout)

        # Define the Net Architecture
        input_img = tfkl.Input(shape=(npix, npix, inchannel))
        x = input_img
        if skipconn:
            skips = []

        # list of all image sizes in the encoding process will be formed,
        # as well as list of needed crops on the way back up
        imagesizes = [npix]
        crops = []

        # save input for later use in residual connection
        # count is number of layers since last res connection
        if resnets:
            count = 0
            addlater = x
            laststr = 1

        for i, feat in enumerate(features):
            # increment number of layers since last res connection
            if resnets:
                count += 1
            pool_here = 0

            # each layer has a specific number of features, depth, and a stride length
            try:
                depth = feat[0]
                strid = feat[1]
                laststr = strid
                if strid != 1:
                    imagesizes.append((imagesizes[-1] - 1) // strid + 1)
                    crops.append(imagesizes[-1] * strid - imagesizes[-2])
                    if max_pool:
                        pool_here = strid
                        strid = 1
                if skipconn:  # state recorded for skip connection any time feature is list
                    skips.append(x)
            except TypeError:  # enables use of list of two numbers or simple number
                depth = feat
                strid = 1

            if dropout > 0:
                x = tfkl.Dropout(dropout)(x)

            # ends of network are allowed different filter sizes than middle
            if i < initfiltersize[1]:
                filtersizehere = initfiltersize[0]
            else:
                filtersizehere = filtersize

            x = tfkl.Conv2D(depth, (filtersizehere, filtersizehere), activation=actfn, padding="same",
                            strides=(strid, strid), kernel_initializer=initialiser,
                            kernel_regularizer=tfk.regularizers.l1(l1reg), use_bias=bias)(x)
            if pool_here:
                x = tfkl.MaxPool2D(pool_here, padding='same')(x)

            # if image size changed, record it.
            if tfk.backend.int_shape(x)[-2] != imagesizes[-1]:
                imagesizes.append(tfk.backend.int_shape(x)[-2])

            # if there have been 2 layers since last res connection,
            # make it, reset the counter, and save state for next one
            if resnets and count == 2:
                if laststr != 1 or tfk.backend.int_shape(addlater)[-1] != depth:
                    addlater = tfkl.Conv2D(depth, (filtersizehere, filtersizehere), activation=None, padding='same',
                                           strides=(laststr, laststr),
                                           kernel_initializer=initialiser,
                                           kernel_regularizer=tfk.regularizers.l1(l1reg),
                                           use_bias=bias)(addlater)
                x = tfkl.Add()([addlater, x])
                count = 0
                laststr = 1
                addlater = x

            if bn:
                x = tfkl.BatchNormalization(axis=3)(x)

        encoded = x
        print("shape of encoded", tfk.backend.int_shape(encoded))
        laststr = 1

        j = 0  # count number of skips we've closed
        k = 0  # count number of crops we've seen (backwards through crops stack)
        for i in range(nl - 1):
            # now we go backwards through the feature specifications given.
            # We always take stride of layer n-i-1, and depth of layer n-i-2:
            # This makes our architecture symmetric about the encoded layer.
            # note we suppose the first layer has stride 1.

            if resnets:
                count += 1

            skiphere = False
            pool_here = 0
            try:
                depth = features[nl - i - 2][0]
            except TypeError:
                depth = features[nl - i - 2]
            try:
                strid = features[nl - i - 1][1]
                laststr = strid
                if strid != 1:
                    k -= 1
                    if max_pool:
                        pool_here = strid
                        strid = 1
                if skipconn:  # if feature is a list, recall state for skip connection
                    skipper = skips[::-1][j]
                    skiphere = True
                    j += 1
            except TypeError:
                strid = 1

            if dropout > 0:
                x = tfkl.Dropout(dropout)(x)

            if i > nl - 1 - initfiltersize[1]:
                filtersizehere = initfiltersize[0]
            else:
                filtersizehere = filtersize

            if pool_here:
                x = tfkl.UpSampling2D(pool_here)(x)

            x = tfkl.Conv2DTranspose(depth, (filtersizehere, filtersizehere), activation=actfn, padding="same",
                                     strides=(strid, strid),
                                     kernel_initializer=initialiser,
                                     kernel_regularizer=tfk.regularizers.l1(l1reg), use_bias=bias)(x)

            if strid != 1 and crops[k]:  # if we need to crop after up stride, do it
                crop = crops[k]
                x = tfkl.Cropping2D(((0, crop), (0, crop)))(x)

            if resnets and count == 2:
                shape1 = tfk.backend.int_shape(addlater)
                shape2 = tfk.backend.int_shape(x)
                if shape1 != shape2:
                    addlater = tfkl.Conv2DTranspose(depth, (filtersizehere, filtersizehere), activation=None,
                                                    padding='same', strides=(laststr, laststr),
                                                    kernel_initializer=initialiser,
                                                    kernel_regularizer=tfk.regularizers.l1(l1reg),
                                                    use_bias=bias)(addlater)
                    if crops[k]:
                        crop = crops[k]
                        addlater = tfkl.Cropping2D(((0, crop), (0, crop)))(addlater)
                x = tfkl.Add()([addlater, x])
                count = 0
                addlater = x
                laststr = 1

            if resnets and i == nl - 2:
                addlatermt = x

            if bn:
                x = tfkl.BatchNormalization(axis=3)(x)

            if skiphere:
                x = tfkl.Concatenate(axis=3)([x, skipper])

        decoded = []
        try:
            depth = features[0][0]
        except TypeError:
            depth = features[0]

        for i in range(outchannel):
            if resnets:
                count = 0
                addlater = addlatermt

            if dropout > 0:
                y = tfkl.Dropout(dropout)(x)
            else:
                y = x
            for j in range(multitasklength):
                if resnets:
                    count += 1

                y = tfkl.Conv2D(depth, (filtersize, filtersize), activation=actfn, padding="same",
                                kernel_initializer=initialiser,
                                kernel_regularizer=tfk.regularizers.l1(l1reg), use_bias=bias)(y)

                if resnets and count == 2:
                    y = tfkl.Add()([addlater, y])
                    count = 0
                    addlater = y

                if bn:
                    y = tfkl.BatchNormalization(axis=3)(y)
                if dropout > 0:
                    y = tfkl.Dropout(dropout)(y)

            y = tfkl.Conv2D(1, (filtersize, filtersize), activation=None, padding="same",
                            kernel_initializer=initialiser, kernel_regularizer=tfk.regularizers.l1(l1reg),
                            use_bias=bias)(y)
            decoded.append(y)

        # print("shape of decoded", decoded.shape)

        self.nnet = tfk.Model(input_img, decoded)
