import os
import tensorflow as tf
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

class VideoDataGenerator(Dataset):
    def __init__(self, data_dir: str, fps: int =22, duration: int =20, height_or_width_pixels: int = 512):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.endswith('.tfrecord') or f.endswith('.mp4')]
        self.num_frames =  fps * duration
        self.height_or_width_size = height_or_width_pixels

    def _parse_tfrecord(self, path):
        frames = []
        raw_dataset = tf.data.TFRecordDataset(path)
        for raw in raw_dataset:
            feat = tf.io.parse_single_example(raw, {'frame': tf.io.FixedLenFeature([], tf.string)})  # JPEG-encoded frame
            frame = tf.io.decode_jpeg(feat['frame'], channels=3)
            frame = tf.image.resize(frame, [self.height_or_width_size, self.height_or_width_size])
            frames.append(frame.numpy().astype(np.uint8))
            if len(frames) == self.num_frames:
                break
        return frames  # the shape is 256x256x3

    def _load_mp4(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.height_or_width_size, self.height_or_width_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        frames = []
        if path.endswith('.mp4'):
            frames = self._load_mp4(path)
        elif path.endswith('.tfrecord'):
            frames = self._parse_tfrecord(path)
        else:
            print("*************Wrong format file present in dataset*************")

        # Pad with zeros if the number of frames is less than num_frames
        if len(frames) < self.num_frames:
            print("the frames in dataset are less than required. this is equivalent to junk. junk in = junk out. so try to avoid this condition")
            zero_pad_frames = [np.zeros((self.height_or_width_size, self.height_or_width_size, 3), dtype=np.float16)] * (self.num_frames - len(frames))
            frames.extend(zero_pad_frames)

        frames = np.stack(frames[:self.num_frames], axis=0)  # Shape: (T, H, W, C)
        video = (frames.astype(np.float16) / 127.5) - 1.0
        video = torch.from_numpy(video).permute(3, 0, 1, 2)
        video_tensor = video.to(torch.bfloat16)
        return {"pixel_values": video_tensor}

# TESTIING CODE
# from torch.utils.data import DataLoader
# tfrecord_dir = "/dataset/dataset_mp4_cleaned/"
# dataset = VideoDataGenerator(tfrecord_dir)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# for batch in dataloader:
#     video = batch['pixel_values']
#     print(f'Video batch shape: {video.shape}')
