# ***sd-webui-udav2***

[Upgraded-Depth-Anything-V2 - UDAV2](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2) extension for a1111. It significantly outperforms [V1](https://github.com/LiheYoung/Depth-Anything) in fine-grained details & robustness. Compared with SD-based models, it enjoys faster inference speed, fewer parameters & higher depth accuracy.

## UDAV2 Extension Preview

![a1111-extension-preview](https://github.com/MackinationsAi/sd-webui-udav2/assets/133395980/f3a98052-8e26-426c-8c5b-a3a22834cfd4)

## News

- **2024-06-14:** Paper, project page, code, models, demo, & benchmark are all released.
- **2024-06-20:** The repo has been upgraded & is also now running on .safetensors models instead of .pth models.
- **2024-06-23:** Updated installation process to be a simpler one_click_install.bat file. It automatically downloads the depth models into a 'checkpoints' folder, the triton wheel into the repo's main folder & installs all of the dependencies needed. *[Also updated this README.md file to provide more clarity!]* [Upgraded-Depth-Anything-V2 - UDAV2](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2)
- **2024-06-24:** [pravdomil](https://github.com/pravdomil) has provided a much need update to UDAV2 for 16bit image creation in order to make stunning 3D Bas-Reliefs! I am currently in the process of updating the gradio webui to include both 16bit single image & 16bit batch image creation which will be pushed in the coming days.
- **2024-06-25:** Working on a beta version of UDAV2 as an a1111 extension & will be released next week, so stay-tuned.
- **2024-06-27:** A1111 extension released!
- **2024-06-29:** Updated forge extension release [sd-forge-udav2](https://github.com/MackinationsAi/sd-webui-udav2/releases/tag/sd-forge-udav2), to prevent conflicts w/ pre-existing installed extensions in Forge! ***OUTDATED***
- **2024-07-01:** Extension has been added to the extension index.json! You can now install this extension directly inside A1111.
- **2024-07-03:** Released v0.0.3 of [sd-forge-udav2](https://github.com/MackinationsAi/sd-webui-udav2/releases/tag/sd-forge-udav2_v0.0.3), meant for anyone using the ***forge-webui***!
- **2024-07-03:** [v1.1.452] [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet/) now has a depth_anything_v2 preprocessor🔥! *Update transformers dependency to [transformers-4.44.1](https://github.com/MackinationsAi/sd-webui-controlnet/releases/tag/transformers-4.44.1) to use the new depth_anything_v2 controlnet preprocessor.*
- **2024-07-24:** Updated upgraded_depth_anything_v2.py script to fix install & usage errors.
- **2024-08-02:** Released v0.0.5 of [sd-forge-udav2](https://github.com/MackinationsAi/sd-webui-udav2/releases/tag/sd-forge-udav2_v0.0.5), updated to prevent depth_anything_v2.dpt NoModule & gradio integration errors w/ Forge.

## Installation

All you need to do is clone this repo into the `stable-diffusion-webui/extensions` folder & then run a1111. *Or* You can install the extension directly from the a1111 extensions tab now! 

```
git clone https://github.com/MackinationsAi/sd-webui-udav2.git
```

## **Important*
If you want to use this extension w/ `stable-diffusion-forge-webui` please install this extension [sd-forge-udav2](https://github.com/MackinationsAi/sd-webui-udav2/releases/tag/sd-forge-udav2_v0.0.3), ***NOT*** `sd-webui-udav2`.

## Notes

- This extension has basically all of the same features & functionality as the main [UDAV2](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2) stand-alone repo w/ some minor changes for the conversion to a1111 extension to work properly.
- The 'Today's Depth Gallery' tab automatically updates after each single image or batch image processing for the current day.
- All of the outputs are saved in a very similar structure to 'txt2img' & 'img2img' in the main a1111 outputs folder under 'depths'
- I'm still making updates to the main which I will push to this extension when possible! *(Including 16bit for hires 3D depth mapping)*
- Bug fixed - ~It usually took between 45-90 secs the first time you launch the webui after installing the extension (as it is downloading the models, once everything is installed you will see in cmd that everything has been installed & will progress w/ the launch).~

## Original DAV2 Github Repo Creds
<div align="center">

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://bingykang.github.io/)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup> · [**Zhen Zhao**](http://zhaozhen.me/) · [**Xiaogang Xu**](https://xiaogang00.github.io/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

Legend <sup>Keys</sup> - [ HKU <sup>1</sup>  ·  TikTok <sup>2</sup>  ·  project-lead &dagger;  ·  corresponding author * ]
</div>

<div align="center">
<a href="https://arxiv.org/abs/2406.09414"><img src='https://img.shields.io/badge/arXiv-Depth Anything V2-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything V2-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>
</div>

![teaser](https://github.com/MackinationsAi/sd-webui-udav2/assets/133395980/c9a277fd-7c3c-4810-949b-4e2bfd3e230c)

## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.

## Citation

If you find this project useful, please consider citing below, give this upgraded repo a star & share it w/ others in the community!

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe & Kang, Bingyi & Huang, Zilong & Zhao, Zhen & Xu, Xiaogang & Feng, Jiashi & Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe & Kang, Bingyi & Huang, Zilong & Xu, Xiaogang & Feng, Jiashi & Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
