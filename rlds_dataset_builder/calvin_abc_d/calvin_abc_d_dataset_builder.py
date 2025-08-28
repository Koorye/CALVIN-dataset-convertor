"""calvin_abc_d dataset."""

import os
os.environ['NO_GCE_CHECK'] = 'true'

import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True

import numpy as np
import tensorflow_datasets as tfds


def get_data(root, idx, start, end, task):
    data = np.load(os.path.join(root, f'episode_{idx:07d}.npz'))
    out = {
        'observation': {
            'image': data['rgb_static'],
            'wrist_image': data['rgb_gripper'],
            'state': data['robot_obs'],
            'environment_state': data['scene_obs'],
        },
        'action': data['actions'],
        'relative_action': data['rel_actions'],
        'discount': 1.0,
        'reward': idx == end,
        'is_first': idx == start,
        'is_last': idx == end,
        'is_terminal': idx == end,
        'language_instruction': task,
    }
    data.close()
    return out


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for calvin_abc_d dataset."""
  
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(200, 200, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(84, 84, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(15,),
                            dtype=np.float64,
                            doc='Robot EEF state (6D pose, gripper open=1/close=-1).',
                        ),
                        'environment_state': tfds.features.Tensor(
                            shape=(24,),
                            dtype=np.float64,
                            doc='Robot joint angles.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot EEF action.',
                    ),
                    'relative_action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot EEF relative action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            'train': self._generate_examples('/home/wushihan/data/Datasets/calvin/task_ABC_D/training'),
            'val': self._generate_examples('/home/wushihan/data/Datasets/calvin/task_ABC_D/validation'),
        }

    def _generate_examples(self, root):
        def _parse_episode(task):
            start, end, lang = task
            sample = {
                'steps': [get_data(root, idx, start, end, lang) for idx in range(start, end + 1)]
            }
            return sample
            
        anno_path = os.path.join(root, 'lang_annotations/auto_lang_ann.npy')
        anno = np.load(anno_path, allow_pickle=True).item()
        tasks = anno['language']['ann']
        task_indexs = anno['info']['indx']
        assert len(tasks) == len(task_indexs)
        tasks = [(task_index[0].item(), task_index[1].item(), task) 
                for task_index, task in zip(task_indexs, tasks)]
        tasks.sort(key=lambda x: x[0])

        print('Total episodes:', len(tasks))

        for idx, task in enumerate(tasks):
            yield f'episode{idx}', _parse_episode(task)
