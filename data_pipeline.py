import os
import tensorflow as tf
import torch
from torch.utils.data import Dataset
import numpy as np

class TFRecordVideoDataset(Dataset):
    def __init__(self, tfrecord_dir: str, num_frames: int = 150):
        self.tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir)
                               if f.endswith('.tfrecord')]
        self.num_frames = num_frames


    def _parse_tfrecord(self, example_proto):
        feature_description = {
            'frame': tf.io.FixedLenFeature([], tf.string)  # JPEG-encoded frame
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        decoded_frame = tf.io.decode_jpeg(parsed_features['frame'], channels=3)
        resized_frame = tf.image.resize(decoded_frame, [256, 256])
        return resized_frame  # the shape is 256x256x3

    def __len__(self):
        return len(self.tfrecord_files)

    def __getitem__(self, idx):
        # Read TFRecord file
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_files[idx])
        frames = []
        for raw_record in raw_dataset:
            frame = self._parse_tfrecord(raw_record).numpy()
            frames.append(frame)
            if len(frames) == self.num_frames:  # Stop if we've collected enough frames
                break
        if len(frames) < self.num_frames:
            padding = [np.zeros((256, 256, 3), dtype=np.float32)] * (self.num_frames - len(frames))
            frames.extend(padding)

        video = np.stack(frames, axis=0)  # Shape: [T, 256, 256, 3]
        video = (video / 127.5) - 1
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # Shape: [T, C, H, W]
        video_tensor = torch.tensor(video, dtype=torch.bfloat16)
        return {"pixel_values": video_tensor}

"""# TESTIING CODE
from torch.utils.data import DataLoader
tfrecord_dir = 'dataset_cleaned'
dataset = TFRecordVideoDataset(tfrecord_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dataloader:
    video = batch['pixel_values']
    print(f'Video batch shape: {video.shape}')"""
