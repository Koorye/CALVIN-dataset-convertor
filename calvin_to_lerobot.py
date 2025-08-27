import numpy as np
import os
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


root = '/home/wushihan/data/Datasets/calvin/task_ABC_D/training'
repo_id = 'Koorye/calvin_abc_d'


def get_data(root, idx):
    data = np.load(os.path.join(root, f'episode_{idx:07d}.npz'))
    out = {
        'observation.images.top': data['rgb_static'],
        'observation.images.wrist': data['rgb_gripper'],
        'observation.state': data['robot_obs'],
        'observation.environment_state': data['scene_obs'],
        'action': data['actions'],
        'relative_action': data['rel_actions'],
    }
    data.close()
    return out


anno_path = os.path.join(root, 'lang_annotations/auto_lang_ann.npy')
anno = np.load(anno_path, allow_pickle=True).item()
tasks = anno['language']['ann']
task_indexs = anno['info']['indx']
assert len(tasks) == len(task_indexs)
tasks = [(task_index[0].item(), task_index[1].item(), task) 
        for task_index, task in zip(task_indexs, tasks)]
tasks.sort(key=lambda x: x[0])

print('Total episodes:', len(tasks))

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=30,
    video_backend='pyav',
    image_writer_processes=5,
    image_writer_threads=10,
    features={
        'observation.images.top': {
            'dtype': 'video',
            'shape': (200, 200, 3),
            'name': ['height', 'width', 'channel'],
        },
        'observation.images.wrist': {
            'dtype': 'video',
            'shape': (84, 84, 3),
            'name': ['height', 'width', 'channel'],
        },
        'observation.state': {
            'dtype': 'float64',
            'shape': (15,),
            'name': [
                'x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                'gripper_width', 
                'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 
                'gripper_action',
            ],
        },
        'observation.environment_state': {
            'dtype': 'float64',
            'shape': (24,),
            'name': [
                'sliding_door', 'drawer', 'button', 'switch', 'lightbulb', 'green_light',
                'red_block.x', 'red_block.y', 'red_block.z', 'red_block.roll', 'red_block.pitch', 'red_block.yaw',
                'blue_block.x', 'blue_block.y', 'blue_block.z', 'blue_block.roll', 'blue_block.pitch', 'blue_block.yaw',
                'pink_block.x', 'pink_block.y', 'pink_block.z', 'pink_block.roll', 'pink_block.pitch', 'pink_block.yaw',
            ],
        },
        'action': {
            'dtype': 'float64',
            'shape': (7,),
            'name': ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
        },
        'relative_action': {
            'dtype': 'float64',
            'shape': (7,),
            'name': ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
        },
    }
)

for idx, task in enumerate(tasks):
    start, end, lang = task
    print(f'[{idx + 1} / {len(tasks)}] Processing episode {start} to {end}, task: {lang}')
    for idx in range(start, end + 1):
        frame = get_data(root, idx)
        dataset.add_frame(frame, task=lang)
    dataset.save_episode()
