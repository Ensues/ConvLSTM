import os
import cv2
import numpy as np
import pandas as pd
import ffmpeg
import os.path as osp
from utils import *

class Preprocessor:
    @staticmethod
    def process_raw_videos(video_ids):
        """Forces videos to 10 FPS so time is consistent."""
        if not os.path.exists(VID_PATH): os.makedirs(VID_PATH)
        
        for vid_id in video_ids:
            input_file = osp.join(VID_PATH_OG, vid_id + ".mp4")
            output_file = osp.join(VID_PATH, vid_id + ".mp4")
            if not os.path.exists(output_file):
                print(f"Standardizing FPS for {vid_id}...")
                ffmpeg.input(input_file).filter('fps', fps=10).output(output_file).run()

    @staticmethod
    def process_video_to_npy(vid_id, sensor_labels):
        """Chops video into .npy frames and matches them to sensor data."""
        vidcap = cv2.VideoCapture(osp.join(VID_PATH, vid_id + ".mp4"))
        video_frames, video_labels = [], []
        ctr = 0

        while True:
            hasFrames, image = vidcap.read()
            if not hasFrames: break

            # 1. Save frame as raw NumPy data
            save_name = f"{vid_id}_{ctr}.npy"
            np.save(osp.join(PROCESSED_PATH, save_name), image)

            # 2. Sync with sensor: Find the label for this timestamp
            # (Repo logic: look 1 second ahead to predict the next move)
            timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC) + 1000
            rounded_ts = timestamp - (timestamp % 100)
            
            label = sensor_labels.get(rounded_ts, 'straight')
            video_labels.append(label_map(label))
            video_frames.append(save_name)
            ctr += 1

        # 3. Save the map
        df = pd.DataFrame({'frames': video_frames, 'labels': video_labels})
        df.to_csv(osp.join(DATA_SAVE_PATH, vid_id + ".csv"), index=None)
        vidcap.release()