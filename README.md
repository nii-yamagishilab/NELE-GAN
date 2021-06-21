# GAN for Near-End Speech Intelligibility Enhancement

Implementation for paper: [Multi-Metric Optimization using Generative Adversarial Networks for Near-End Speech Intelligibility Enhancement](https://arxiv.org/abs/2104.08499)

[Audio samples](https://nii-yamagishilab.github.io/hyli666-demos/intelligibility)

## Usage steps

### 1. Dependencies

#### 1). Intelligibility and Quality metrics

* SIIB: https://github.com/kamo-naoyuki/pySIIB
* STOI (ESTOI): https://github.com/mpariente/pystoi
* PESQ: https://github.com/vBaiCai/python-pesq
* ViSQOL: https://github.com/google/visqol
* HASPI: We provide an unofficial Python-based HASPI implementation in ***pyHASPI*** folder. 
  Note that this implementation is not exactly same with the original one. Please check ***pyHASPI/README.txt***. For more details on HASPI, please check the following reference papers:

    [1]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI)." Speech Communication 65 (2014): 75-93.
    
    [2]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI) Version 2." Speech Communication (2020).
    
#### 2). Another important dependencies:
* python==3.7
* librosa==0.7.1
* numpy==1.17.2
* torch==1.2.0
* matplotlib==3.1.1

### 2. Prepare training data

Prepare your training data

For data format, a toy dataset example is given in ***./toy_dataset*** 

Note: I normalized all training utterances into RMS=0.03 for convenient processing, but it is not mandatory.

### 3. Training

Run: `python train_nele.py`

You should modify training configurations according to your need, `e.g. data path, GAN_epoch, num_of_sampling...`

models will be saved in ***./chkpt*** 

### 4. Inference

Run: `python inference.py`

A pre-trained model is stored in  ***./trained_model/chkpt_GD.pt***  <br/>It was trained using 16 kHz speech materials at RMS=0.03. So please normalize your 16kHz raw speech input to RMS=0.03, if you would like to use this pre-trained model.

---

## Authors
* Haoyu Li
* [Junichi Yamagishi](https://nii-yamagishilab.github.io/)

## Acknowlegment

This work was partially supported by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan, and by MEXT KAKENHI Grants (16H06302, 17H04687, 18H04120, 18H04112, 18KT0051), Japan. 


This project was partially based on [MetricGAN](https://github.com/JasonSWFu/MetricGAN) codes.

IMCRA noise estimation algorithm was revised from [Observation Uncertainty tools](https://github.com/ramon-astudillo/obsunc/blob/e849aac65a16fe6900061505fbb0e30f594bd99a/processing/imcra.py)




---
## License

BSD 3-Clause License

Copyright (c) 2021, Yamagishi and Echizen Laboratories, National Institute of Informatics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
