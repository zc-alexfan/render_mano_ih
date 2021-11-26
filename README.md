# MANO Renderer

> A quick-and-dirty script to render hand part segmentation masks from MANO meshes in [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M). The segmentations are used in [DIGIT](https://github.com/zc-alexfan/digit-interacting). 

Before installing, check if you have CUDA 10.1 compiler. If not, the troubleshooting section might be useful.

```
which nvcc
nvcc --version
> /usr/local/cuda-10.1/bin/nvcc
```

## Setting Up Environment

The code was tested on PyTorch 1.6.0, Python 3.6.12, Ubuntu 20.04.  

```
git clone https://github.com/zc-alexfan/render_mano_ih.git
cd render_mano_ih
git clone https://github.com/zc-alexfan/neural_renderer
conda create -n render.mano python=3.6.12
conda activate render.mano
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
cd neural_renderer
python setup.py install
```

## Troubleshooting
- [Unsupported GNU version! gcc 8 and up are not supported!](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version)

```
MAX_GCC_VERSION=8
sudo apt install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION
sudo ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/local/cuda-10.1/bin/gcc
sudo ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/local/cuda-10.1/bin/g++
```

- [Install CUDA 10.1 (nvcc) in Ubuntu 20.04](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0). You may need to reboot.

## Prepare files

```
cd render_mano_ih
mkdir -p data/InterHand2.6M
cd data/meta_data
```

Download `J_regressor_mano_ih26m.npy` to `./data/meta_data/` from here:

```
https://github.com/facebookresearch/InterHand2.6M/blob/1f11fe90f52bc5205173e07dd3adfe048a8546a9/tool/MANO_world_to_camera/J_regressor_mano_ih26m.npy
```

Follow the instruction of [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M) and download its 5fps.v1 of the dataset. Under `./data/InterHand2.6M`, put the images and annotation.

The files `MANO_*.pkl` are the MANO models of SMPLX; you can obtain them [here](https://github.com/vchoutas/smplx).

The file structure should be like this:
```
tree ./data
|-- InterHand2.6M
|   |-- annotations
|   `-- images
`-- meta_data
    |-- J_regressor_mano_ih26m.npy
    |-- model
    |   |-- MANO_LEFT.pkl
    |   `-- MANO_RIGHT.pkl
    `-- seale_faces.npy
```

Inside `./data/annotations` it should look like the below:
```
tree ./data/annotations
|-- skeleton.txt
|-- subject.txt
|-- train
   |-- InterHand2.6M_train_MANO_NeuralAnnot.json
   |-- InterHand2.6M_train_camera.json
   |-- InterHand2.6M_train_data.json
   `-- InterHand2.6M_train_joint_3d.json
```

Images should be in a structure like this:
`./data/InterHand2.6M/images/train/Capture0/0012_aokay_upright/cam400002/*.jpg`

## Render

```
# render training segm masks
python main.py

# package segm masks into LMDB for DIGIT
python package_segm_lmdb.py
``` 

## Acknowledgement
- The original code was from [neural renderer](https://github.com/daniilidis-group/neural_renderer) but we use the version from [adambielski](https://github.com/adambielski/neural_renderer).
- [Muhammed Kocabas](https://github.com/mkocabas) created the original scripts for rendering part segmentation for human body in [PARE](https://github.com/mkocabas/PARE).
- We modified the rendering code from [PARE](https://github.com/mkocabas/PARE) to allow MANO rendering.
- The original part segmentation of hands is from [Jinlong Yang](https://ps.is.tuebingen.mpg.de/person/jyang).

If you found the code useful, please consider citing:

```bibtex

@inproceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}

@inproceedings{Kocabas_PARE_2021,
  title = {{PARE}: Part Attention Regressor for {3D} Human Body Estimation},
  author = {Kocabas, Muhammed and Huang, Chun-Hao P. and Hilliges, Otmar and Black, Michael J.},
  booktitle = {Proc. International Conference on Computer Vision (ICCV)},
  pages = {11127--11137},
  month = oct,
  year = {2021},
  doi = {},
  month_numeric = {10}
}

@inproceedings{fan2021digit,
  title={Learning to Disambiguate Strongly Interacting Hands via Probabilistic Per-pixel Part Segmentation},
  author={Fan, Zicong and Spurr, Adrian and Kocabas, Muhammed and Tang, Siyu and Black, Michael and Hilliges, Otmar},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```









