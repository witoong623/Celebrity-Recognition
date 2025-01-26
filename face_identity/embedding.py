import argparse
from pathlib import Path

import numpy as np
from deepface import DeepFace
from sqlalchemy import select

from config import Config
from db_models import Face, get_session_maker


def get_face_embedding(filepath: str, config: Config, is_face_image=True) -> np.ndarray | None:
    # by default, don't detect face in image
    represent_args = {
        'enforce_detection': False,
        'detector_backend': 'skip'
    }
    if not is_face_image:
        represent_args['detector_backend'] = config.face_detector_backend
        represent_args['enforce_detection'] = True

    try:
        embedding_obj = DeepFace.represent(
            img_path=filepath,
            model_name=config.embedding_model,
            # normalization use the same name as the model
            normalization=config.embedding_model,
            **represent_args
        )
    except ValueError:
        return None

    # embedding is list of D float values where D is embedding size
    # originally, shape is (1, D), dtype is float32
    return np.array(embedding_obj[0]['embedding'], dtype=np.float32).reshape(1, -1)
