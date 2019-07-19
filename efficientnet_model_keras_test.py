import json
import os
import pickle
from typing import List, Dict

import keras
import numpy as np
import tensorflow as tf

from convert_weight import dump_params
from efficientnet_model_keras import EfficientNetModelBuilder, EfficientNetBlockParams, EfficientNetGlobalParams, \
    EfficientNetParams, conv_kernel_initializer, Swish, dense_kernel_initializer

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def read_data(data_path=os.path.join(BASE_PATH, 'data/cifar-10-batches-py'), count: int = None):
    import pickle
    X, Y = [], []
    for i in range(1, 6):
        data_and_label = pickle.loads(open(os.path.join(data_path, 'data_batch_1'), 'rb').read(), encoding='bytes')
        data: np.ndarray = data_and_label[b'data']
        labels: List[int] = data_and_label[b'labels']

        X.append(np.transpose(np.reshape(data, (data.shape[0], 3, 32, 32)), (0, 2, 3, 1)))
        Y.append(np.array(labels))
    if count is not None:
        X, Y = np.concatenate(X)[:count], np.concatenate(Y)[:count]
    else:
        X, Y = np.concatenate(X), np.concatenate(Y)

    from eval_ckpt_main import MEAN_RGB, STDDEV_RGB
    X = np.array(X, dtype=np.float32)
    X -= np.array(MEAN_RGB, dtype=np.float32).reshape((1, 1, 3))
    X /= np.array(STDDEV_RGB, dtype=np.float32).reshape((1, 1, 3))
    return X, Y


def load_weights(model: keras.Model, pickle_weights_dir: str):
    vars_dict: Dict[str, np.ndarray] = pickle.loads(open('{}/{}.params.pickle'.format(pickle_weights_dir, model.name), 'rb').read())
    print('weights count in pickle: ', len(vars_dict))
    layers: List[keras.layers.Layer] = [layer for layer in model.layers if layer.weights]
    print('weights count in model: ', sum([len(layer.weights) for layer in layers]))

    print(list(vars_dict.keys()))
    for layer in layers:
        print('layer: {}, names: {}'.format(layer.name, [w.name for w in layer.weights]))

    from convert_weight import map_weight_key
    weight_value_tuples = []
    for layer in layers:
        for w in layer.weights:
            print(w.name)
            key_in_pickle = map_weight_key(model.name, w.name)
            weight_value_tuples.append((w, vars_dict[key_in_pickle]))
    keras.backend.batch_set_value(weight_value_tuples)


def train():
    import keras
    X, Y = read_data(count=10)
    print(Y)
    model_name = 'efficientnet-b0'
    model = EfficientNetModelBuilder().build(model_name, input_shape=(32, 32, 3), num_classes=10)
    model.summary()
    opti = keras.optimizers.RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(opti,
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    load_weights(model, 'data')
    # model.load_weights()
    model.fit(X, Y, batch_size=2)
    model.save_weights(os.path.join(BASE_PATH, 'data/keras_model_saved_weights/{}.h5'.format(model_name)))


def test_load_weights():
    from eval_ckpt_main import MEAN_RGB, STDDEV_RGB
    X = pickle.loads(open('data/x.pickle', 'rb').read())
    X = np.array(X, dtype=np.float32)
    X -= np.array(MEAN_RGB, dtype=np.float32).reshape((1, 1, 3))
    X /= np.array(STDDEV_RGB, dtype=np.float32).reshape((1, 1, 3))

    model_name = 'efficientnet-b0'
    model = EfficientNetModelBuilder().build(model_name, input_shape=(224, 224, 3), num_classes=1000)
    model.summary()
    # model.load_weights('data/converted_weights/{}_imagenet_1000.h5'.format(model_name))
    load_weights(model, 'data')
    Y = model.predict(X)
    # keras.utils.plot_model(model, './data/model.png')
    # model.save_weights('data/converted_weights/{}_imagenet_1000.h5'.format(model_name))
    print(Y.tolist())
    from keras_model import softmax, print_eval_result
    pred_probs = softmax(Y)
    pred_idx = np.argsort(pred_probs)[:, ::-1]
    pred_prob = np.array([pred_probs[i][pid] for i, pid in enumerate(pred_idx)])[:, :5]
    pred_idx = pred_idx[:, :5]

    img_file = 'data/panda.jpg'
    print_eval_result(pred_idx, pred_prob, [img_file], 'data/labels_map.txt')


def demo_model():
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.optimizers import SGD

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=[keras.metrics.sparse_categorical_accuracy], optimizer=sgd)

    X, Y = read_data(count=10)
    model.fit(X, Y, batch_size=32, epochs=10)


def lib_model():
    from efficientnet import EfficientNetB0
    model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    for layer in model.layers:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    x = model.outputs[0]
    _global_params = EfficientNetGlobalParams.get(EfficientNetParams.for_model('efficientnet-b0'), num_classes=10)
    batch_norm_momentum = _global_params.batch_norm_momentum
    batch_norm_epsilon = _global_params.batch_norm_epsilon
    channel_axis = 1 if _global_params.channel_first else -1
    x = keras.layers.Conv2D(
        name='head_conv',
        filters=EfficientNetBlockParams.round_filters_based_on_depth_multiplier(1280, _global_params),
        kernel_size=[1, 1], strides=[1, 1], padding='same',
        use_bias=False, kernel_initializer=conv_kernel_initializer)(x)
    x = keras.layers.BatchNormalization(
        name='head_bn', axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
    x = Swish(name='head_act')(x)

    x = keras.layers.GlobalAveragePooling2D(name='head_pooling', data_format=_global_params.data_format)(x)
    x = keras.layers.Dense(name='head_dense', units=_global_params.num_classes, kernel_initializer=dense_kernel_initializer)(x)
    model = keras.models.Model(model.inputs, x)

    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy], optimizer=sgd)

    X, Y = read_data(count=10)
    model.fit(X, Y, batch_size=32, epochs=10)


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=1)
    return softmax_x


def test_run_tf_model():
    import pickle
    x = pickle.loads(open('data/x.pickle', 'rb').read())[0]

    with tf.Graph().as_default(), tf.Session() as sess:
        model_name = 'efficientnet-b0'

        X = tf.cast(tf.stack([x]), dtype=tf.float32)
        X -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=X.dtype)
        X /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=X.dtype)

        with tf.variable_scope(model_name):
            blocks_args, global_params = efficientnet_builder.get_model_params(model_name, None)
            model = efficientnet_model.Model(blocks_args, global_params)
            _logits = model(X, False)
            model.summary()

        sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.latest_checkpoint('data/models/{}'.format(model_name))
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
        for v in tf.global_variables():
            if 'moving_mean' in v.name or 'moving_variance' in v.name:
                ema_vars.append(v)
        ema_vars = list(set(ema_vars))
        var_dict = ema.variables_to_restore(ema_vars)
        print(list(var_dict.keys()))

        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, checkpoint)

        logits = model.predict(X, steps=1, batch_size=1)

    pred_probs = softmax(logits)
    pred_idx = np.argsort(pred_probs)[:, ::-1]
    pred_prob = np.array([pred_probs[i][pid] for i, pid in enumerate(pred_idx)])[:, :5]
    pred_idx = pred_idx[:, :5]

    classes = json.loads(open('data/labels_map.txt', 'r').read())
    print('predicted class for image {}: '.format('data/panda.jpg'))
    for i, idx in enumerate(pred_idx[0]):
        print('  -> top_{} ({:4.2f}%): {}  '.format(i, pred_prob[0][i] * 100, classes[str(idx)]))


if __name__ == '__main__':
    # train()
    # demo_model()
    # lib_model()
    dump_params()
    test_load_weights()
