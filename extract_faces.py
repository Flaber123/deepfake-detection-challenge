import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1

from dfdc.video import VideoReader
from dfdc.image import isotropically_resize_image, make_square_image

# set paths
data_path = Path('D:/Data/deepfake-detection-challenge')
train_path = data_path / 'dfdc_train_all'
train_dirs = os.listdir(train_path)


def save_image(data, filename):
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, dpi=sizes[0])
    plt.close()


# set settings
# B1: image_size=240, margin=40
# B3: image_size=300, margin=70
verbose = False
fake_frames = 2
real_frames = 5
input_size = 240
video_reader = VideoReader()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=240,
              margin=80,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7],
              factor=0.709,
              post_process=False,
              select_largest=False,
              keep_all=False,
              device=device)
print('Running on device: {}'.format(device))

# TODO: Implement dataframe with names, paths, labels, and probabilities
metadata = pd.DataFrame(columns=['video', 'chunk', 'path', 'frame', 'label'])

for chunk in train_dirs[:1]:
    start_chunk = time.time()
    input_dir = train_path / chunk
    output_dir = data_path / 'dfdc_processed' / chunk
    labels = pd.read_json(str(input_dir / 'metadata.json'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created new directory: {}'.format(output_dir))

    videos = [f for f in os.listdir(input_dir) if (os.path.isfile(os.path.join(input_dir, f))) and (f.endswith('.mp4'))]
    num_videos = len(videos)

    for video, video_idx in zip(videos, range(1, num_videos + 1)):
        # capture video
        start_video = time.time()
        video_path = input_dir / video
        video_capture = cv2.VideoCapture(str(video_path))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        video_label = labels[video].iloc[0]
        if video_label == 'FAKE':
            num_frames = fake_frames + 2  # subtract first and last frame afterwards
        elif video_label == 'REAL':
            num_frames = real_frames + 2  # subtract first and last frame afterwards
        else:
            raise ValueError

        frame_indices = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
        frame_indices = frame_indices[1: -1]
        valid_indices = []
        frames = []

        # loop trough videos and get frames
        for frame_idx in range(frame_count):
            grab = video_capture.grab()  # load frame
            if not grab:
                print('Error grabbing frame {}} from video {}'.format(frame_idx, video_path))
                break  # stop processing video

            if frame_idx in frame_indices:
                retrieve, frame = video_capture.retrieve()
                if (not retrieve) or (frame is None):
                    print('Error retrieving frame {}} from video {}'.format(frame_idx, video_path))
                    break
                else:
                    valid_indices.append(frame_idx)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))

        # detect faces in batch
        # TODO: Batch saving to file: https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
        faces = mtcnn(frames)

        # save face images
        for face, frame_idx in zip(faces, frame_indices):
            if face is not None:
                output_file = video.rstrip('.mp4') + '_{}_{}.png'.format(str(frame_idx).zfill(3), video_label)
                output_file = output_dir / output_file
                save_image(face.permute(1, 2, 0).int().numpy(), output_file)
                if verbose:
                    print('Saved face in frame {} of video: {}'.format(frame_idx, video))
            else:
                continue

        # finish of video
        end_video = round(time.time() - start_video, 1)
        print('Finished video {} of {} in {}s'.format(video_idx, num_videos, end_video))

    # finish of chunk
    end_chunk = round(time.time() - start_chunk, 1)
    print('\n', 'Finished chunk {} of {} in {}s'.format(1, 50, end_chunk), '\n')
