import re
import tensorflow as tf


MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def map_weight_key(model_name: str, keras_key: str) -> str:
    key = keras_key\
        .replace('stem_conv', 'conv2d').replace('stem_bn', 'tpu_batch_normalization')\
        .replace('head_conv', 'conv2d').replace('head_bn', 'tpu_batch_normalization').replace('head_dense', 'dense')\
        .replace('block_1_1_depthwise_bn', 'tpu_batch_normalization').replace('block_1_1_depthwise_conv', 'depthwise_conv2d')\
        .replace('block_1_1_project_bn', 'tpu_batch_normalization_1').replace('block_1_1_project_conv', 'conv2d')\
        .replace('block_1_1_se_reduce_conv', 'conv2d').replace('block_1_1_se_expand_conv', 'conv2d_1')\
        .replace(':0', '')

    match = re.match(r'.*block_(\d)_(\d)_', keras_key)
    if match is not None:
        block_idx, sub_block_idx = tuple(map(int, match.groups()))
        block_prefix = 'block_{}_{}'.format(block_idx, sub_block_idx)
        if not (block_idx == 1 and sub_block_idx == 1):
            key = key.replace('{}_expand_bn'.format(block_prefix), 'tpu_batch_normalization').replace('{}_expand_conv'.format(block_prefix), 'conv2d')\
                .replace('{}_depthwise_bn'.format(block_prefix), 'tpu_batch_normalization_1').replace('{}_depthwise_conv'.format(block_prefix), 'depthwise_conv2d')\
                .replace('{}_se_reduce_conv'.format(block_prefix), 'conv2d').replace('{}_se_expand_conv'.format(block_prefix), 'conv2d_1')\
                .replace('{}_project_bn'.format(block_prefix), 'tpu_batch_normalization_2').replace('{}_project_conv'.format(block_prefix), 'conv2d_1')\

    return '{}/{}/ExponentialMovingAverage'.format(model_name, key)


def read_data():
    import pickle
    x = pickle.loads(open('data/x.pickle', 'rb').read())[0]
    X = tf.stack([x])
    X = tf.cast(X, dtype=tf.float32)
    X -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=X.dtype)
    X /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=X.dtype)
    return X


def dump_params(model_name='efficientnet-b0'):
    with tf.Graph().as_default(), tf.Session() as sess:
        X = read_data()

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
        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, checkpoint)

        vars = dict([(name, sess.run(var)) for name, var in var_dict.items()])

    import pickle
    pickle.dump(vars, open('data/{}.params.pickle'.format(model_name), 'wb'))
