import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import shutil
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

class VideoDataset(Dataset):


    r"""A Dataset for a folder of videos. Expects the directory structure to be
    RGB_video (directory)->video.mp4. RGB_video.txt lists all video name and labels. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.



        Args:
            dataset (str): Name of dataset.
            train_index / val_index (list) : Index of the train index. Got from StratifiedKFold.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 15.
            process (bool): Determines whether to preprocess dataset. Default is False.
    """



    def __init__(self, split, fnames, labels, type_video, clip_len=16, process=True):
        # root_dir = '../../RGB_video' output_dir = '../../RGB_frames' label_path='../../RGB_label.txt'
        self.root_dir, self.output_dir, self.label_path = Path.db_dir(type_video)
        self.clip_len = clip_len
        self.fnames = fnames
        self.labels = labels
        self.type_video = type_video
        self.crop_size = 112
        self.resize_height=112
        self.resize_width = 112
        self.split = split


        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' )

        if process:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            if split == 'train':
                    os.mkdir(os.path.join(self.output_dir, 'train'))
            elif split == 'val':
                    os.mkdir(os.path.join(self.output_dir, 'val'))
            else:
                    os.mkdir(os.path.join(self.output_dir, 'test'))
            for video in self.fnames:
                self.process_video(video, split, os.path.join(self.output_dir, split))



        assert len(self.labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))



    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(os.path.join( self.output_dir, self.split, self.fnames[index]))
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.labels[index])

        if self.split == 'train':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        # print(buffer.shape)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True



    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video
        # video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % 1 == 0:
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):

        if self.type_video  == "gray":
            mean = np.array([0.485])  # Adjust mean and std for grayscale images
            std = np.array([0.229])
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        for i, frame in enumerate(buffer):
            frame = (frame - mean) / std
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        # if self.type_video == "gray":
        #     return buffer
        # else:
            return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):

        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            if self.type_video == "gray":
                frame = cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)  # Load grayscale image
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            frame = cv2.resize(frame, (self.resize_height, self.resize_width))
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)
        height_index, width_index =0 , 0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


def get_label_and_video_name(path):
    """read label txt file to extract video name and label
    """
    label = []
    video_name = []
    data = open(path, 'r')
    data_lines = data.readlines()

    for data_line in data_lines:
        data = data_line.strip().split(' ')
        video_name.append(data[0])
        if data[1] == 'solid':
            label.append(0)
        elif data[1] =='half':
            label.append(1)
        else:
            label.append(2)
    return label, video_name


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    fnames,labels= get_label_and_video_name(Path.db_dir('gray')[2])[1],get_label_and_video_name(Path.db_dir('gray')[2])[0]


    for train_index, val_index in skf.split(X=fnames, y=labels):
        
        fnames_train, labels_train =[fnames[i] for i in train_index], [labels[i] for i in train_index]


        train_data = VideoDataset(split='train', fnames=fnames_train, labels=labels_train,type_video='gray', clip_len=16, process=True)
        #  def __init__(self,train_index, val_index, split, fnames, labels, type_video='gray', clip_len=10, process=True):
        train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=4)

        shutil.rmtree(os.path.join(Path.db_dir('gray')[1],'train'))

        # for i, sample in enumerate(train_loader):
        # 
        #     inputs = sample[0]
        #     labels = sample[1]
        #     print(inputs.size())
        #     print(labels)
        # 
        #     if i == len(train_loader) - 1:  # 检查是否为最后一个批次
        #         # 调整最后一个批次的批次大小
        #         last_batch_size = len(train_loader) % train_loader.batch_size
        #         break

