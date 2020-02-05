import numpy as np
import tensorflow as tf


class TFDataProcesser:
    """Generating data structure for tensorflow training"""

    def __init__(self, batch_size, buffer_size, drop_remainder=False):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder

    def data2tf_data_set(self, features, labels, is_large_numpy=False):
        """Note that if tensors contains a NumPy array, and eager execution is not enabled, the values will be
        embedded in the graph as one or more tf.constant operations. For large datasets (> 1 GB), this can waste memory
        and run into byte limits of graph serialization. If tensors contains one or more large NumPy arrays, please set
        the parameter as:True.
        Tips:If tensors contains one or more large NumPy arrays, please set the parameter as:True.Must initialize firstly
        before your loop code
        """
        if not is_large_numpy:
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            dataset = dataset.batch(batch_size=self.batch_size,
                                    drop_remainder=self.drop_remainder).repeat(2)
            try:
                dataset_iter = dataset.make_one_shot_iterator()
            except:
                dataset_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
            iteration_element = dataset_iter.get_next()
            return dataset, iteration_element
        else:
            # #=============tensors contains one or more large NumPy arrays===============
            # data slices
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            # shuffle batched repeat
            dataset = dataset.shuffle(self.buffer_size).batch(batch_size=self.batch_size,
                                                              drop_remainder=self.drop_remainder).repeat()
            iterator = dataset.make_initializable_iterator()
            iteration_element = iterator.get_next()
            return iterator, iteration_element

    def tfrecord2tf_data_set(self, tf_files, feature_description):
        dataset = tf.data.TFRecordDataset(filenames=[tf_files])
        dataset = dataset.shuffle(self.buffer_size).map(self._parse_function).batch(batch_size=self.batch_size,
                                                                                    drop_remainder=self.drop_remainder).repeat()
        dataset = dataset.make_one_shot_iterator()
        iteration_element = dataset.get_next()
        return iteration_element

    def _parse_function(self, exampled_string, feature_description):
        """feature_description = {
            'name' : tf.io.FixedLenFeature([], tf.string, default_value='Nan'),
            'label': tf.io.FixedLenFeature([] , tf.int64, default_value=-1)
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data' : tf.io.FixedLenFeature([], tf.string)
        }"""
        return tf.io.parse_single_example(exampled_string, feature_description)


if __name__ == '__main__':
    features = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    labels = [1, 2, 3, 4, 5]

    EPOCHS = 20
    BATCH_SIZE = 2
    BUFFER_SIZE = 10
    NUM_BATCHES = 3
    is_large_numpy = False
    processer = TFDataProcesser(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    data_iteration = processer.data2tf_data_set(features=features, labels=labels, is_large_numpy=is_large_numpy)
    if tf.__version__.split('.')[0] == 1:
        with tf.Session() as sess:
            for epoch in range(EPOCHS):
                print("==============Start epoch:{}===============".format(epoch))
                for batch in range(NUM_BATCHES):
                    value = sess.run(data_iteration)
                    print(value)
                print("==============Finish epoch:{}!===============".format(epoch))
        """++++++++++++++++++Large numpy condition++++++++++++++++++++"""
        is_large_numpy = True
        features = np.array(features)
        labels = np.array(labels)
        # build dataset
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        processer = TFDataProcesser(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
        iterator, data_element = processer.data2tf_data_set(features=features, labels=labels,
                                                            is_large_numpy=is_large_numpy)
        with tf.Session() as sess:
            # must initialize before loop
            sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                                      labels_placeholder: labels})
            for epoch in range(EPOCHS):
                print("==============Start epoch:{}===============".format(epoch))
                for batch in range(NUM_BATCHES):
                    value = sess.run(data_iteration)
                    print(value)
                print("==============Finish epoch:{}!===============".format(epoch))
    else:
        for epoch in range(EPOCHS):
            print("==============Start epoch:{}===============".format(epoch))
            for data in data_iteration:
                # TODO just one line data, looping func get every line after batch.Need to modify in tensorflow 2
                print(data)
            print("==============Finish epoch:{}!===============".format(epoch))
        """++++++++++++++++++Large numpy condition++++++++++++++++++++"""
        is_large_numpy = True
        features = np.array(features)
        labels = np.array(labels)
        # build dataset
        features_placeholder = tf.compat.v1.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
        processer = TFDataProcesser(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
        iterator, data_element = processer.data2tf_data_set(features=features, labels=labels,
                                                            is_large_numpy=is_large_numpy)
        with tf.compat.v1.Session() as sess:
            # must initialize before loop
            sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                                      labels_placeholder: labels})
            for epoch in range(EPOCHS):
                print("==============Start epoch:{}===============".format(epoch))
                for batch in range(NUM_BATCHES):
                    value = sess.run(data_iteration)
                    print(value)
                print("==============Finish epoch:{}!===============".format(epoch))
