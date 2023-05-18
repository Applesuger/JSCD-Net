from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, Activation, BatchNormalization, \
    Dropout, add, GlobalAveragePooling3D, \
    Reshape, Dense, Permute, multiply, GlobalMaxPooling3D, Lambda, AveragePooling3D,Cropping3D
from keras.optimizers import RMSprop
from keras.regularizers import l2
import tensorflow as tf

from group_norm import GroupNormalization
from nn.metrics import dice_coefficient,focal_loss, create_add_dice,create_weighted_binary_crossentropy


K.set_image_data_format("channels_last")
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

smooth = 1.
dropout_rate = 0.5
act = "relu"

def standard_unit(input_tensor, stage, nb_filter):
    x = Conv3D(nb_filter, 3, activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x1 = Conv3D(nb_filter, 3, activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x1 = Dropout(dropout_rate, name='dp'+stage+'_2')(x1)
    return x1


def squeeze_excite_block(input, stage, ratio=8):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
               name='sed1_' + stage)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
               name='sed2_' + stage)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x


def Conv3d_BN(x, nb_filter, kernel_size, stage, strides=1, padding='same'):
    x = Conv3D(nb_filter, kernel_size, padding=padding, data_format='channels_last', strides=strides,
               activation='relu', name='conv_' + stage)(x)
    x = BatchNormalization(name='bn_' + stage)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, stage, strides=1, with_conv_shortcut=False):
    x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same',
                  stage=stage + '_c1')
    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same', stage=stage + '_c2')
    x = CBAM_block(x,nb_filter=nb_filter)
    if with_conv_shortcut:
        shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size, stage=stage + '_shortcut')
        x = Dropout(0.2, name=stage + '_d1')(x)
        x = add([x, shortcut], name=stage + '_a1')
        return x
    else:
        x = add([x, inpt])
        return x

def InceptionV3(input_tensor,stage):
    branch_a = Conv3D(8, 1, strides=1, name=stage+'b1')(input_tensor)

    branch_b = Conv3D(8, 1, strides=1, padding='same', name=stage+'b2_1')(input_tensor)
    branch_b = Conv3D(8, 3, strides=1, padding='same', name=stage+'b2_2')(branch_b)

    branch_c = AveragePooling3D(3, strides=1, padding='same', name=stage+'b3_1')(input_tensor)
    branch_c = Conv3D(4, 3, strides=1, padding='same', name=stage+'b3_2')(branch_c)

    branch_d = Conv3D(8, 1, strides=1, padding='same', name=stage+'b4_1')(input_tensor)
    branch_d = Conv3D(12, 3, strides=1, padding='same', name=stage+'b4_2')(branch_d)
    branch_d = Conv3D(12, 3, strides=1, padding='same', name=stage+'b4_3')(branch_d)

    output = concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1, name=stage+'c1')
    output = Activation('relu', name=stage+'a1')(GroupNormalization(groups=16, name=stage+'g1')(output))
    return output


def CBAM_block(x,nb_filter,reduction_ratio=16):

    # Channel Attention
    avgpool = GlobalAveragePooling3D()(x)
    maxpool = GlobalMaxPooling3D()(x)
    # Shared MLP
    Dense_layer1 = Dense(nb_filter//reduction_ratio, activation='relu')
    Dense_layer2 = Dense(nb_filter, activation='relu')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = add([avg_out, max_out])
    channel = Activation('sigmoid')(channel)
    channel = Reshape((1, 1, 1, nb_filter))(channel)
    channel_out = multiply([x, channel])

    # Spatial Attention
    avgpool = Lambda(lambda x:tf.reduce_mean(x, axis=-1, keepdims=True))(channel_out)
    maxpool = Lambda(lambda x:tf.reduce_max(x, axis=-1, keepdims=True))(channel_out)
    spatial = concatenate([avgpool, maxpool],axis=-1)

    spatial = Conv3D(1, (7, 7, 7), strides=1, padding='same')(spatial)
    spatial_out = Activation('sigmoid')(spatial)

    CBAM_out = multiply([channel_out, spatial_out])

    return CBAM_out

def UNetPlusPlus(input_shape,num_class=1,deep_supervision = True):

        nb_filter = [32, 64, 128, 256, 512]

        x = input_shape

        # x00 - x10
        conv1_1 =standard_unit(x, stage='11', nb_filter=nb_filter[0])

        pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)
        # x10 - x20
        conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
        pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)
        # x10 - x01
        up1_2 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up12', padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=-1)   #conv1_1_cbam
        conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])
        # x20 - x30
        conv3_1= standard_unit(pool2, stage='31', nb_filter=nb_filter[2])

        pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)
        # x20 - x11
        up2_2 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up22', padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=-1)   #conv2_1_cbam
        conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

        # x11 - x02
        up1_3 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up13', padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1,conv1_2], name='merge13', axis=-1)  # x02 x00 x01 conv1_2_cbam
        conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])
        # x30
        conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])


        # x30 - x21
        up3_2 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up32', padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=-1)  # x21 x20conv3_1_cbam
        conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

        # x21 - x12
        up2_3 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up23', padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=-1)  # x12 x10 x11conv2_2_cbam
        conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])
        # x12 - x03
        up1_4 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up14', padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=-1)  # x03 x00 x01 x02conv1_3_cbam
        conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

        """Segmentation network output 64*64*64"""
        nestnet_output_1 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_1',
                                  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
        nestnet_output_2 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_2',
                                  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
        nestnet_output_3 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_3',
                                  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)


        if deep_supervision:
            model = Model(input=x, output=[nestnet_output_1,
                                           nestnet_output_2,
                                           nestnet_output_3,
                                            ])
        else:
            model = Model(input=x, output=[nestnet_output_3])
        # when unet++ pre-training,
        # model.compile(optimizer=Adam(lr=1e-4, decay=1e-5), loss=create_add_dice,
        #               metrics=['binary_crossentropy', 'accuracy', dice_coefficient])

        return model

def CLSNet(input_shape,nestnet_output_3):
    nb_filter = [32, 64, 128, 256, 512]

    x1 = input_shape
    x2 = Cropping3D(cropping=((16, 16), (16, 16), (16, 16)))(x1)
    x3 = Cropping3D(cropping=((8, 8), (8, 8), (8, 8)))(x2)

    inception1_1 = InceptionV3(x1,stage='11')

    conv1 = Conv3d_BN(inception1_1, 32, 1, stage="11", strides=1)

    conv2_seg = Conv3d_BN(nestnet_output_3, 32, 1, stage="12", strides=1)
    con_seg_cls = concatenate([conv1, conv2_seg], axis=-1)

    inception2_1 = InceptionV3(x2,stage="21")
    conv2 = Conv3d_BN(inception2_1, 64, 1, stage="21", strides=1)

    inception3_1 = InceptionV3(x3,stage="31")
    conv3 = Conv3d_BN(inception3_1, 128, 1, stage="31", strides=1)

    ib1 = identity_Block(con_seg_cls, nb_filter=nb_filter[1], kernel_size=(3, 3, 3), stage='ib4', strides=2,
                         with_conv_shortcut=True)
    ib1_add = add([ib1,conv2])
    ib2 = identity_Block(ib1_add, nb_filter=nb_filter[2], kernel_size=(3, 3, 3), stage='ib5', strides=2,
                         with_conv_shortcut=True)
    ib2_add = add([ib2,conv3])
    ib3 = identity_Block(ib2_add, nb_filter=nb_filter[3], kernel_size=(3, 3, 3), stage='ib6', strides=2,
                         with_conv_shortcut=True)

    con1_cls = GlobalAveragePooling3D(data_format='channels_last', name='con4')(ib1)
    con2_cls = GlobalAveragePooling3D(data_format='channels_last', name='con5')(ib2)
    con3_cls = GlobalAveragePooling3D(data_format='channels_last', name='con6')(ib3)

    con4_cls = concatenate([con1_cls, con2_cls, con3_cls], axis=-1, name='merge')

    out_class = Dense(1, activation='sigmoid', name='d1_2')(con4_cls)
    model_cls = Model(input = x1,output = out_class)
    return model_cls

def merge_model(input_shape):
    x = Input(input_shape)
    model_seg = UNetPlusPlus(x)
    model_cls = CLSNet(x,model_seg.output[2])
    # load unet++ pre-training weights document
    model_seg.load_weights('E:/aneurysmDetection_segment/data/unetplus_deep/****.h5')

    for layer in model_seg.layers:
           layer.trainable = False

    segout = model_seg.output
    clsout = model_cls.output
    model = Model(input=x,output=[segout[0],segout[1],segout[2],clsout])

    model.compile(optimizer=RMSprop(lr=0.0001), loss={
        'output_1': create_add_dice,
        'output_2': create_add_dice,
        'output_3': create_add_dice,
        'd1_2':  focal_loss(0.25,2)
    },
                  loss_weights={
                      'output_1': 1,
                      'output_2': 1,
                      'output_3': 1,
                      'd1_2': 1
                  },
                  metrics={ 'output_1': ['accuracy', dice_coefficient],
                      'output_2': ['accuracy', dice_coefficient],
                      'output_3':['accuracy', dice_coefficient],
                      'd1_2': 'accuracy'
                  })
    return model