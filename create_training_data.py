import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN


class FaceExtractor:
    def __init__(self, data_path, image_size=240, margin=80, fake_frames=2, real_frames=5, verbose=False):
        # path settings
        self.data_path = Path(data_path)
        self.train_path = self.data_path / 'dfdc_train_all'
        self.train_dirs = os.listdir(self.train_path)

        # TODO: Check if file already exists, load if this is the case
        self.metadata = pd.DataFrame(columns=['video', 'chunk', 'path', 'image_size',
                                              'margin', 'frame', 'label', 'target'])

        # image settings
        # B1: image_size=240, margin=80
        # B3: image_size=300, margin=70
        self.image_size = image_size
        self.margin = margin
        self.fake_frames = fake_frames
        self.real_frames = real_frames
        self.verbose = verbose
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.mtcnn = MTCNN(image_size=self.image_size,
                           margin=self.margin,
                           min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7],
                           factor=0.709,
                           post_process=False,
                           select_largest=False,
                           keep_all=False,
                           device=self.device)

        if self.verbose:
            print('Running on device: {}'.format(self.device))

    def set_mtcnn(self, image_size=240, margin=80, min_face_size=20, select_largest=False, keep_all=False):
        self.mtcnn = MTCNN(image_size=image_size,
                           margin=margin,
                           min_face_size=min_face_size,
                           thresholds=[0.6, 0.7, 0.7],
                           factor=0.709,
                           post_process=False,
                           select_largest=select_largest,
                           keep_all=keep_all,
                           device=self.device)

    @staticmethod
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

    def run(self):
        for chunk, chunk_idx in zip(self.train_dirs[45:], range(45, len(self.train_dirs) + 1)):
            start_chunk = time.time()
            input_dir = self.train_path / chunk
            output_dir = self.data_path / 'dfdc_processed' / chunk
            labels = pd.read_json(str(input_dir / 'metadata.json'))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print('Created new directory: {}'.format(output_dir))

            videos = [f for f in os.listdir(input_dir) if (os.path.isfile(os.path.join(input_dir, f))) and
                      (f.endswith('.mp4'))]
            num_videos = len(videos)

            for video, video_idx in zip(videos, range(1, num_videos + 1)):
                # TODO: Check if video already in metadata, skip if this is the case

                # capture video
                start_video = time.time()
                video_path = input_dir / video
                video_capture = cv2.VideoCapture(str(video_path))
                frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                video_label = labels[video].iloc[0]
                if video_label == 'FAKE':
                    num_frames = self.fake_frames + 2  # subtract first and last frame afterwards
                    target = 1
                elif video_label == 'REAL':
                    num_frames = self.real_frames + 2  # subtract first and last frame afterwards
                    target = 0
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
                        # print('Error grabbing frame {}} from video {}'.format(frame_idx, video_path))
                        break  # stop processing video

                    if frame_idx in frame_indices:
                        retrieve, frame = video_capture.retrieve()
                        if (not retrieve) or (frame is None):
                            # print('Error retrieving frame {}} from video {}'.format(frame_idx, video_path))
                            break
                        else:
                            valid_indices.append(frame_idx)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(Image.fromarray(frame))

                # detect faces in batch
                faces = self.mtcnn(frames)

                # save face images
                for face, frame_idx in zip(faces, frame_indices):
                    if face is not None:
                        output_file = video.rstrip('.mp4') + '_{}_{}.png'.format(str(frame_idx).zfill(3), video_label)
                        output_file = output_dir / output_file
                        self.save_image(face.permute(1, 2, 0).int().numpy(), output_file)

                        # update dataframe containing metadata
                        meta = {'video': video, 'chunk': chunk, 'path': output_file, 'image_size': self.image_size,
                                'margin': self.margin, 'frame': frame_idx, 'label': video_label, 'target': target}
                        self.metadata = self.metadata.append(meta, ignore_index=True)
                    else:
                        continue

                # finish of video
                end_video = round(time.time() - start_video, 2)
                print('Finished video {}/{} in {}s'.format(video_idx, num_videos, end_video))

            # finish of chunk
            self.metadata.to_csv(str(self.data_path / 'dfdc_processed/metadata.csv'), index=False)
            end_chunk = round(time.time() - start_chunk, 2)
            print('Finished chunk {} of {} in {}s'.format(chunk_idx, 50, end_chunk), '\n')


if __name__ == "__main__":
    # run face extractor pipeline
    face_extractor = FaceExtractor('D:/Data/deepfake-detection-challenge', 240, 80, 1, 3, True)
    face_extractor.run()
