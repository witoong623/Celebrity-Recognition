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