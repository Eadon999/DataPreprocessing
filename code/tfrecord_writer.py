import time
import tensorflow as tf
from code.data_preprocessing import TFDataProcesser


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
            try:
                # tensorflow version < 2.0.0
                _options = tf.python_io.TFRecordOptions(tf.python.python_io.TFRecordCompressionType.ZLIB)
            except:
                _options = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.ZLIB)

        elif compress_options == 'gzip':
            try:
                _options = tf.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
            except:
                _options = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        else:
            _options = None
        print("start write tfrecord...")
        try:
            writer = tf.python_io.TFRecordWriter(self.path, options=_options)
        except:
            writer = tf.compat.v1.python_io.TFRecordWriter(self.path, options=_options)
        features_external = tf.train.Features(feature=self.feature_internal_dict)
        # To avoid error:'TypeError: No positional arguments allowed',tf.train.Example must explicate the para: features
        exampled = tf.train.Example(features=features_external)
        exampled_serialized = exampled.SerializeToString()
        writer.write(exampled_serialized)
        writer.close()
        print("write data to tfrecord finished! Cost time (ms):{}".format(time.time() - start_time))


if __name__ == '__main__':
    width = 1
    weights = 0.1
    values = "0.1 0.2 0.1 0.3".encode('utf-8')
    feature_internal = {
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "values": tf.train.Feature(bytes_list=tf.train.BytesList(value=[values])),
        "weights": tf.train.Feature(float_list=tf.train.FloatList(value=[weights]))
    }
    writer = TfrecordWriter(dst='../output/test.tfrecord', feature_internal=feature_internal)
    writer.write2tfrecord()
    data_processer = TFDataProcesser(1, 222)
    feature_description = {
        'width' : tf.io.FixedLenFeature([], tf.int64),
        'weights' : tf.io.FixedLenFeature([], tf.int64)
    }
    data_processer.tfrecord2tf_data_set("../output/test.tfrecord", feature_description)
