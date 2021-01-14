import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator

from tensorflow.python.feature_column.feature_column import _LazyBuilder

"""
numeric_column(
    key,
    shape=(1,),
    default_value=None,
    dtype=tf.float32,
    normalizer_fn=None
)
key: 特征的名字。也就是对应的列名称。
shape: 该key所对应的特征的shape. 默认是1，但是比如one-hot类型的，shape就不是1，而是实际的维度。总之，这里是key所对应的维度，不一定是1.
default_value: 如果不存在使用的默认值
normalizer_fn: 对该特征下的所有数据进行转换。如果需要进行normalize，那么就是使用normalize的函数.这里不仅仅局限于normalize，也可以是任何的转换方法，比如取对数，取指数，这仅仅是一种变换方法.
"""


def test_numeric():
    price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x + 2

    price_column = tf.feature_column.numeric_column('price', normalizer_fn=transform_fn)

    price_transformed_tensor = price_column._get_dense_tensor(builder)

    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

    # 使用input_layer

    price_transformed_tensor = tf.feature_column.input_layer(price, [price_column])

    with tf.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor]))
    """
    [array([[ 3.],
           [ 4.],
           [ 5.],
           [ 6.]], dtype=float32)]
    use input_layer________________________________________
    [array([[ 3.],
           [ 4.],
           [ 5.],
           [ 6.]], dtype=float32)]
    从上面的结果可以看出，transform_fn 将所有的数值+2来处理了。使用_LazyBuilder和inpu_layer来分别进行了测试.效果是一样的
    """
"""
bucketized_column(
    source_column,
    boundaries
)
source_column: 必须是numeric_column
boundaries: 不同的桶。boundaries=[0., 1., 2.],产生的bucket就是, (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf), 每一个区间分别表示0, 1, 2, 3,所以相当于分桶分了4个.
"""
def test_bucketized_column():

    price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本

    price_column = tf.feature_column.numeric_column('price')
    bucket_price = tf.feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])

    price_bucket_tensor = tf.feature_column.input_layer(price, [bucket_price])

    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))
    """[array([[ 0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.]], dtype=float32)]
       我们看到分桶之后，会直接转换成one-hot形式的
    """


if __name__ == '__main__':
    test_numeric()
    test_bucketized_column()