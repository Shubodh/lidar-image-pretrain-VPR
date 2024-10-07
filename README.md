# LiDAR Image Pretraining for Cross Modal Visual Place Recognition

### Paper's Crisp Summary: What is the deal?

We called upon Master Yoda to explain it succinctly:     
For the domains of image and language, the original [CLIP](https://openai.com/research/clip) was, hmm. Apply this pretraining to the domains of LiDAR and image, we do, specifically for the task of cross-modal place recognition, yes. Here, [the link to our paper](https://www.shubodhs.ai/liploc), you will find. Enjoy this code you must, hehe.

### Code's Crisp Summary: How this code could be used?
The code is written well in a modular way so that this can be used for many applications beyond the scope of the paper.
1. You are just exploring LiDAR Image Pretraining in general for Robotics or Visual Place Recognition applications.
2. Replicate the results of our paper and build on cross modal localization methods.
3. Expand beyond LiDAR, 2D images and incorporate other domains like language.

## Setting up the environment
This code is tested on Ubuntu 22.04 with Python 3.7 with PyTorch 1.13.1 and CUDA 11.6 with following packages. But this is a general PyTorch deep learning setup and should work with any modern existing setup/configuration with Python 3. Just a few misc packages can be installed using pip or conda.

Main packages:
pytorch, torchvision, numpy, scipy, matplotlib, opencv-python  

Misc packages:
* timm (0.9.2), tyro (0.5.3), tqdm, albumentations (1.3.0)
* pyyaml, pillow, h5py, scikit-image, scikit-learn


You can directly install above or using the provided `requirements.txt` file using `pip install -r requirements.txt`. However, for PyTorch and torchvision, please install them according to specific CUDA version directly from https://pytorch.org. These lines have been commented out in `requirements.txt`.

After installation of the above packages, activate the environment, then run the commands in "Code" section below.

## Dataset
### KITTI-360:
Download from https://www.cvlibs.net/datasets/kitti-360/download.php.   
Download these two:
1. Perspective Images for Train & Val (128G)  
2. Raw Velodyne Scans (119G)

### KITTI:
Download from https://www.cvlibs.net/datasets/kitti/eval_odometry.php   
Download: velodyne laser data (80 GB), color (65 GB).

After downloading, set the path to the dataset in the config file as mentioned in next section.


## Config files
Corresponding to our paper Section "4. Experiments and Results", we have provided the config files with same name in the `./configs` folder. You just need to set the dataset path in the config file (examples: `exp_combined.py`, `exp_360.py`) to replicate our results using `data_path` or `data_path_360`.


## Code with arguments for specific experiments

Training and evaluation commands are as follows. You can change the arguments as per your requirement.   

### For KITTI:
--expid is the experiment name, for example exp_large, exp_combined etc. (You can replicate as per naming convention used in the paper. Same convention used here.)    
--eval_sequence is the sequence number for evaluation: 08 and 09    
--threshold_dist is the distance threshold for evaluation: we used 20 everywhere.   
1. Train:  
    1. `python trainer.py --expid exp_large`  
2. Evaluate:  
    1. `python evaluate.py --expid exp_large --eval_sequence 08 --threshold_dist 20`  

### For KITTI-360:
1. Train:  
    1. `python trainer.py --expid exp_360`  
2. Evaluate:  
    1. `python evaluate.py --expid exp_360 --eval_sequence 0000 --threshold_dist 20`

## Pretrained models release: Download trained models to run evaluation directly

You can also choose to download the trained models from the following link and run the evaluation script on them directly.
https://drive.google.com/drive/folders/1kpWmchrC8LYXORL8N30xRQprP8AxjHSB?usp=drive_link


## Cite Our Work
Thank you for interest in our work. If our paper or code is useful to you, kindly cite it as:

```
@inproceedings{shubodh2024lip,
  title={Lip-loc: Lidar image pretraining for cross-modal localization},
  author={Shubodh, Sai and Omama, Mohammad and Zaidi, Husain and Parihar, Udit Singh and Krishna, Madhava},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={948--957},
  year={2024}
}

```

Developers:   
* [Sai Shubodh Puligilla](https://www.shubodhs.ai/)   
* [Mohammad Omama](https://mohdomama.github.io/)   
* [Husain Zaidi](https://husainhz7.github.io/)   
* [Udit Singh Parihar](https://udit.netlify.app/)   
