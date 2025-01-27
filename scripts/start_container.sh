#!/bin/bash

docker run --rm -it \
	-v $(pwd):/Celebrity-Recognition \
	--name celebrity-recognition-container \
	--workdir /Celebrity-Recognition \
	celebrity-recognition:latest \
	/bin/bash
