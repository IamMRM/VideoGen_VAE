import tensorflow as tf
import numpy as np

def parse_tfrecord(example_proto):
    feature_description = {
        "frames": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    frames = tf.io.decode_raw(example["frames"], tf.uint8)
    frames = tf.reshape(frames, [600, 256, 256, 3])  # Adjust shape to your data
    return frames

def ltx_data_generator(tfrecord_path, batch_size=4, chunk_size=65):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    
    def split_into_chunks(frames):
        # Split 600 frames into chunks of 65 frames with overlap
        chunks = []
        for i in range(0, 600 - chunk_size + 1, chunk_size):
            chunk = frames[i:i+chunk_size]
            chunk = (tf.cast(chunk, tf.float32) / 127.5) - 1.0  # Normalize to [-1, 1]
            chunks.append(chunk)
        return tf.data.Dataset.from_tensor_slices(chunks)
    
    dataset = dataset.flat_map(split_into_chunks)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Usage
dataset = ltx_data_generator("dataset_cleaned/your_file.tfrecord")
for batch in dataset:
    # batch shape: [B, 65, 256, 256, 3]
    # First frame is conditioning image (batch[:, 0])
    # Remaining 64 frames are targets (batch[:, 1:])
    # Pass to model training here