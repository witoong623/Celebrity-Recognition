# Celebrity Recognition

## Dataset structure
This repository expect the dataset in the following format.
```
dataset/
├── train/
│   ├── 001/
│   │   ├── 001_0001.jpg
│   │   ├── 001_0002.jpg
│   │   └── 001_0003.jpg
│   ├── 002/
│   │   ├── 002_0001.jpg
│   │   ├── 002_0002.jpg
│   │   └── 002_0003.jpg
│   └── 003/
│       ├── 003_0001.jpg
│       ├── 003_0002.jpg
│       └── 003_0003.jpg
└── test/
   ├── 001/
   │   ├── 001_0004.jpg
   │   └── 001_0005.jpg
   ├── 002/
   │   ├── 002_0004.jpg
   │   └── 002_0005.jpg
   └── 003/
       ├── 003_0004.jpg
       └── 003_0005.jpg
```
One must have this dataset directory inside this project root directory, otherwise you must edit the `scripts/start_container.sh` script to mount the dataset directory manually.

## Environment
One can run all steps to demo face recognition using Docker container. Please follow the following steps to build and start Docker container.
1. Run the following script to build docker image, the image will be tagged as `celebrity-recognition:latest`.
```bash
./scripts/build_docker_image.sh
```
2. Run the following script to start docker container. After you run, you will be inside the container. If you exit the container, you need to start container again (start script contains `--rm` option)
```bash
./scripts/start_container.sh
```
After this point, one must run everything from inside container.

## Configuration
One can config parameters to run python scripts onwards. The configuration file is `config.yaml`. Every python script accepts `--config` which is path to YAML config file, it defaults to `config.yaml`, so if you want to use new config file, you must use `--config new-yaml-config.yaml` in every python script.

You may change the following configs:
- `db_file`: name of SQLite DB file.
- `dataset_dir`: path to the root dataset directory. **You need to change this if you use difference name from the recommended structure above.**
- `preprocessed_dataset_dir`: path to the cropped face images.

**You shouldn't change other configs** because it ties to the available implementation of the Deepface that this repository use.

## Building face embedding database
In order to use face recognition, one must build the database of known celebrity face. The following are steps to preprocess and build known face embedding database
1. Detect face in all images, then crop and then write to DB and crop face image and save to filesystem.
```
python face_identity/detect_face.py
```
2. Extract face embedding and save to DB.
```
python face_identity/build_face_embedding.py
```
3. Filter out outlier face images from DB.
```
python face_identity/filter_outlier_image.py
```

After this, you will have face embedding ready to be used for building face embedding vector DB.

## Evaluate celebrity recognition performance
Run the following command to see recognition system performance test with testing dataset.
```
python face_identity/evaluate.py
```

## Running demo script
Use `face_identity/demo.py` script to run the system to recognize celebrity. You need to provide path the image file.

Optionally, you can provide threshold using `--threshold` to help determine whether it can recognize or not. By default, the sytem will recognize to the most similar celebrity.

If it can't detect face, it can't recognize. Example of running `face_identity/demo.py` script.
```
python face_identity/demo.py --image dataset/test/1725014/1725014-0034.jpg
```

## Notebook
There are Jupyter notebook that I used for visualize something in `notebook` directory. If you want to run it, you may need to change the filepath in config.yaml to be absolute path.
