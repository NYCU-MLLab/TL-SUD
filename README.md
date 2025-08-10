## Requirements
- Python 3.10 or above
- Pytorch 2.5 or above
- NVIDIA GPU

## Conda / Pip
```bash
conda create --name nemo python==3.10.12
pip install "nemo_toolkit[asr]"
```
## Training
```bash
python speech_to_text_tlo.py
python tlsud_post.py
```
