# SoftVC VITS Singing Voice Conversion

## Model Overview
A singing voice coversion (SVC) model, using the SoftVC encoder to extract features from the input audio, sent into VITS along with the F0 to replace the original input to acheive a voice conversion effect. Additionally, changing the vocoder to [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to fix the issue with unwanted staccato.

## Notice
* Note: 3.0 models are not compatible with 4.0.
* This is a fork of the 4.0 branch of so-vits-svc. It implements
  the same inference GUI as found in the `eff` branch of this
  repository, with a few extra features relating to 4.0 models
  (such as automatic pitch prediction and clustering). For instructions on using the GUI see the `eff` [branch](https://github.com/effusiveperiscope/so-vits-svc/tree/eff)

### 4.0 Features
+ Feature input replaced with [Content Vec](https://github.com/auspicious3000/contentvec) 
+ Sampling rate changed to 44100hz
+ Due to change of parameters such as hop size and simplification of model structures，VRAM usage for inference has been greatly reduced compared to version 3.0.
+ Code refactor
+ Dataset production and training process are consistent with 3.0; however, models and preprocessed data are not compatible.
+ Added automatic pitch f0 prediction for voice conversion (will be out of tune if used with singing voices)
+ Reduced timbre leakage through k-means clustering scheme

Demo：[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/innnky/sovits4)

## Required downloads
+ Download ContentVec model:[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + Place under `hubert`.
+ Download pretrained models [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/G_0.pth) and [D_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/D_0.pth)
  + Place under `logs/44k`.
  + Pretrained models are required, because from experiments, training from scratch can be rather unpredictable to say the least, and training with a pretrained model can greatly improve training speeds.
  + The pretrained model includes云灏, 即霜, 辉宇·星AI, 派蒙, and 绫地宁宁, covering the common ranges of both male and female voices, and so it can be seen as a rather universal pretrained model.

```shell
wget -P logs/44k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/G_0.pth
wget -P logs/44k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/D_0.pth
```

## Colab notebook scripts

[Colab training notebook (EN)](https://colab.research.google.com/drive/1laRNiMSgSw_SxSnuti8oWIuC--RHzAGp?usp=sharing)

[Colab inference notebook (EN)](https://colab.research.google.com/drive/1128nhe0empM7u4uo5hbZx5lqjgjG1OSf?usp=sharing)

Note that the following notebooks are not maintained by me.

[![Colab training notebook (CN)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19fxpo-ZoL_ShEUeZIZi6Di-YioWrEyhR#scrollTo=0gQcIZ8RsOkn)

## Dataset preparation
All that is required is that the data be put under the `dataset_raw` folder in the structure format provided below.
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## Data pre-processing.
1. Resample to 44100hz

```shell
python resample.py
 ```
2. Automatically sort out training set, validation set, test set, and automatically generate configuration files.
```shell
python preprocess_flist_config.py
```
3. Generate hubert and F0 features/
```shell
python preprocess_hubert_f0.py
```
After running the step above, the `dataset` folder will contain all the pre-processed data, you can delete the `dataset_raw` folder after that.

## Training.
```shell
python train.py -c configs/config.json -m 44k
```
Note: The old model will be automatically cleared during training, and only the latest 5 models will be kept. If you want to prevent overfitting, you need to manually back up the model record points, or modify the configuration file keep_ckpts 0 to never clear.

To train a cluster model, train a so-vits-svc 4.0 model first (as above), then execute `python cluster/train_cluster.py`.

## Inference
For instructions on using the GUI see the `eff` [branch](https://github.com/effusiveperiscope/so-vits-svc/tree/eff)
Otherwise use [inference_main.py](inference_main.py)
Command line support has been added for inference

```shell
# Example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

Required fields
+ -m, --model_path: model path
+ -c, --config_path: configuration file path
+ -n, --clean_names: list of wav file names placed in `raw` folder
+ -t, --trans: pitch transpose (semitones)
+ -s, --spk_list: target speaker names

Optional fields
+ -a, --auto_predict_f0:Automatic pitch prediction; do not enable when converting singing or it will be out of tune.
+ -cm, --cluster_model_path:Path of cluster model
+ -cr, --cluster_infer_ratio:Ratio of clustering to use

# Optional fields
### Automatic f0 prediction
The 4.0 model training process will train an f0 predictor. For voice conversion
you can enable automatic pitch prediction. Do not enable this function when
converting singing voices unless you want it to be out of tune.
### Cluster timbre leakage
Clustering is used to make the model trained more like the target timbre at the cost of articulation/intelligibility. The model can linearly control the proportion of non-clustering scheme (more intelligible, 0) vs. clustering scheme (more speaker-like, 1).

## Onnx export
Use [onnx_export.py](onnx_export.py)
+ Create a new folder:`checkpoints` and open it
+ Create a new folder in the `checkpoints` folder and name it after your project such as `aziplayer`
+ Rename your model to `model.pth`，rename the config file to `config.json`，and place it in the project folder (`aziplayer` )
+ In [onnx_export.py](onnx_export.py) change `path = "NyaruTaffy"` to your project name e.g. `path = "aziplayer"`
+ Run [onnx_export.py](onnx_export.py) 
+ After execution is completed，A `model.onnx` file will be generated in your project folder, which is the exported model
### Onnx UI
   + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)


