# CALVIN Dataset Convertor

Convert CALVIN Dataset to LeRobot & RLDS format.

## Download

Prepare:
```bash
pip install modelscope # modelscope
pip install huggingface-hub # huggingface
```

CALVIN ABC-D LeRobot:
```bash
modelscope download Koorye/calvin-abc-d-lerobot --repo-type dataset # modelscope
TODO # huggingface
```

CALVIN ABC-D RLDS:
```bash
modelscope download Koorye/calvin-abc-d-rlds --repo-type dataset # modelscope
TODO # huggingface
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download raw CALVIN dataset:
```bash
wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
```

2. Convert to LeRobot:
```bash
python calvin_to_lerobot.py
```

3. Convert to RLDS:
```bash
cd rlds_dataset_builder
tfds build calvin_abc_d
```