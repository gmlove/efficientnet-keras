import unittest

from convert_weight import map_weight_key


class ConvertWeightTest(unittest.TestCase):

    @unittest.skip('integration test')
    def test_convert_weight_for_efficient_b0(self):
        keras_weight_keys = open('data/keras_model_saved_weights/efficientnet-b0.weight_keys', 'r').readlines()
        keras_weight_keys = [line.strip() for line in keras_weight_keys]
        tf_weight_keys = open('data/models/efficientnet-b0.weight_keys', 'r').readlines()
        tf_weight_keys = [line.strip() for line in tf_weight_keys]
        model_name = 'efficientnet-b0'
        k_layer_and_keys = [(line.split(': ')[0], line.split(': ')[1][2:-2].split('\', \'')) for line in
                            keras_weight_keys]
        print(k_layer_and_keys)
        assert len(tf_weight_keys) == sum([len(layer_weight_keys) for _, layer_weight_keys in k_layer_and_keys])
        converted_keys = set()
        for layer, layer_weight_keys in k_layer_and_keys:
            for k_key in layer_weight_keys:
                tf_key = map_weight_key(model_name, k_key)
                print('{}: {} -> {}'.format(layer, k_key, tf_key))
                assert tf_key in tf_weight_keys, '{} should in {}'.format(tf_key, tf_weight_keys)
                assert tf_key not in converted_keys, '{} should not in {}'.format(tf_key, converted_keys)
                converted_keys.add(tf_key)
