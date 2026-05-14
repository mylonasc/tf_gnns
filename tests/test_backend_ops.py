import tensorflow as tf

from tf_gnns import backend_ops
from tf_gnns.graphnet_utils import unsorted_segment_max_or_zero, unsorted_segment_min_or_zero


def test_segment_mean_matches_tf_unsorted_segment_mean():
    values = tf.constant([[1.0, 3.0], [2.0, 5.0], [9.0, -1.0]], dtype=tf.float32)
    indices = [0, 0, 2]
    num_groups = 4

    got = backend_ops.segment_mean(values, indices, num_groups)
    expected = tf.math.unsorted_segment_mean(values, tf.constant(indices), num_groups)
    tf.debugging.assert_near(got, expected, atol=1e-6, rtol=1e-6)


def test_segment_zero_semantics_match_existing_functions():
    values = tf.constant([[3.0], [7.0]], dtype=tf.float32)
    indices = [0, 2]
    num_groups = 4

    got_min = backend_ops.segment_min_or_zero(values, indices, num_groups)
    got_max = backend_ops.segment_max_or_zero(values, indices, num_groups)

    expected_min = unsorted_segment_min_or_zero(values, tf.constant(indices), num_groups)
    expected_max = unsorted_segment_max_or_zero(values, tf.constant(indices), num_groups)
    tf.debugging.assert_equal(got_min, expected_min)
    tf.debugging.assert_equal(got_max, expected_max)
