import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1

from dfdc.video import VideoReader
from dfdc.image import isotropically_resize_image, make_square_image

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


num_frames = 10
input_size = 224
video_reader = VideoReader()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7],
              factor=0.709,
              post_process=True,
              select_largest=True,
              keep_all=False,
              device=device)
print('Running on device: {}'.format(device))

for chunk in train_dirs[:1]:
    input_dir = train_path / chunk
    output_dir = data_path / 'processed' / chunk

    labels = pd.read_json(str(input_dir / 'metadata.json'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created new directory: {}'.format(output_dir))

    videos = [f for f in os.listdir(input_dir) if (os.path.isfile(os.path.join(input_dir, f))) and (f.endswith('.mp4'))]
    num_videos = len(videos)

    for video in videos[:10]:
        video_path = input_dir / video
        # frames, idx = video_reader.read_frames(path=str(video_path), num_frames=num_frames)

        # Loop through video, taking a handful of frames to form a batch
        v_cap = cv2.VideoCapture(str(video_path))
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for i in range(v_len):
            success = v_cap.grab()  # load frame
            if i % 50 == 0:
                success, frame = v_cap.retrieve()
            else:
                continue

            if not success:
                continue

            # add to batch
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        # Detect faces in batch
        faces = mtcnn(frames)

        for i in range(len(faces)):
            # plt.figure(figsize=(1,1))
            # plt.imshow(face.permute(1, 2, 0).int().numpy())
            # plt.axis('off')
            # plt.show()

            output_file = video.rstrip('.mp4') + '_{}.png'.format(str(i).zfill(2))
            output_file = output_dir / output_file

            face = faces[i]
            save_image(face.permute(1, 2, 0).int().numpy(), output_file)

        # for frame, index in zip(frames[:1], idx[:1]):
        #     # detect face
        #     face = mtcnn(frame)
        #     face = face.permute(1, 2, 0).int().numpy()  # reshape tensor
        #
        #     # visualize face
        #     if face is not None:
        #         output_file = video.rstrip('.mp4') + '_{}.png'.format(str(index).zfill(2))
        #         output_file = output_dir / output_file
        #
        #         img = Image.fromarray(face)
        #         img.save(output_file)
        #         # img.show()
        #
        #         # plt.imshow(face)
        #         # plt.axis('off')
        #         # plt.savefig(str(output_file))
        #         # plt.show()
        #         # plt.close()
        #         print('Saved face for frame {} of {}'.format(index, video))
        #
        #     else:
        #         print('Did not find a face for video: {}'.format(video))
