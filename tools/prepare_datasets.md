# Getting Started

## Dataset Preparation

Please place the data (RefCOCO, RefCOCO+, RefCOCOg, ReferItGame) or the soft link of dataset folder under ./ln_data/. We follow dataset structure DMS. To accomplish this, the [download_data.sh](../ln_data/download_data.sh) bash script from DMS can be used.

```
cd ./ln_data
bash download_data.sh --path .
```

Please download data indices from [[Gdrive]](https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view?usp=sharing) or [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/???), and place them as the ./data folder.

Need to move them to ./ln_data!!!

```
rm -r ./data
tar -xvf data.tar
```