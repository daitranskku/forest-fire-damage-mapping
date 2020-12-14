from tensorflow.keras.models import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input


def create_model(img_height=912, img_width=912):
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    num_class = 1

    inputs = Input(shape=(img_height, img_width, 3), name='main_input')

    conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        inputs)
    conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        pool1)
    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12',
                            padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_2)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_2)

    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        pool2)
    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22',
                            padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_2)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_2)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13',
                            padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13',
                          axis=bn_axis)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_3)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_3)

    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        pool3)
    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32',
                            padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_2)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_2)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23',
                            padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23',
                          axis=bn_axis)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_3)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_3)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14',
                            padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14',
                          axis=bn_axis)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_4)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_4)

    #     conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(
        pool4)
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(
        conv5_1)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42',
                            padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        conv4_2)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        conv4_2)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33',
                            padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33',
                          axis=bn_axis)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_3)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_3)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24',
                            padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24',
                          axis=bn_axis)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_4)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_4)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15',
                            padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4],
                          name='merge15', axis=bn_axis)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_5)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_5)

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_5)

    model = Model(inputs, nestnet_output_4)

    return model