# CasFT
CasFT leverages observed information Cascades and dynamic cues modeled via neural ODEs as conditions to guide the generation of Future popularity-increasing Trends through a diffusion model.
- Jing, X., Jing, Y., Lu, Y., Deng, B., Chen, X., & Yang, D*. (2024). CasFT: Future Trend Modeling for Information Popularity Prediction with Dynamic Cues-Driven Diffusion Models.
In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2025.

## How to run the code

### Environment

```shell
# create virtual environment
conda create --name casft python=3.7
# activate environment
conda activate casft
# install pytorch==1.12.0
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# install other requirements
pip install -r requirements.txt
```

### Run the code
```shell
#Run at the root of this repo:
python setup.py build_ext --inplace
#preprocess
bash preprocess.sh
#train
bash run.sh
```

## Datasets
Datasets download link: [Google Drive](https://drive.google.com/file/d/1dGdIzyFiRVBsdTek2x5r7s7-fqKrJA_q/view?usp=drive_link)

Put the 'dataset.txt' files (twitter/weibo/aps ) into the corresponding directory: such as 'data/twitter/dataset.txt'

The datasets we used in the paper are come from:

- [Twitter](https://drive.google.com/file/d/1dGdIzyFiRVBsdTek2x5r7s7-fqKrJA_q/view?usp=drive_link )
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019).  

## Reference
If you use our code or datasets, please cite:
```
@inproceedings{jing2024casft,
  title={CasFT: Future Trend Modeling for Information Popularity Prediction with Dynamic Cues-Driven Diffusion Models},
  author={Jing, Xin and Jing, Yichen and Lu, Yuhuan and Deng, Bangchao and Chen, Xueqin and Yang, Dingqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={},
  year={2025}
}
```

