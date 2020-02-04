import time
import tensorflow as tf


class TfrecordWriter:
    def __init__(self, dst, feature_internal):
        """

        :param dst:
        :param feature_internal:
        tf.train.BytesList(value=[value]) # value to binary list
        tf.train.FloatList(value=[value]) # value to float list
        tf.train.Int64List(value=[value]) # value to int list

        """
        self.path = dst
        self.feature_internal_dict = feature_internal

    def write2tfrecord(self, compress_options=None):
        start_time = time.time()
        if compress_options == 'zlib':
            _options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        elif compress_options == 'gzip':
            _options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        else:
            _options = None
        print("start write tfrecord...")
        writer = tf.python_io.TFRecordWriter(self.path, options=_options)
        features_external = tf.train.Features(self.feature_internal_dict)
        exampled = tf.train.Example(features_external)
        exampled_serialized = exampled.SerializeToString()
        writer.write(exampled_serialized)
        writer.close()
        print("write data to tfrecord finished! Cost time (ms):{}".format(time.time() - start_time))


if __name__ == '__main__':
    width = 0.1
    weights = [1,2,3]
    image_raw = "train"
    feature_internal = {
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "weights": tf.train.Feature(float_list=tf.train.FloatList(value=[weights])),
        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    }
