import argparse
from pathlib import Path

import numpy as np
from deepface import DeepFace
from sqlalchemy import select

from config import get_config
from db_models import Face, get_session_maker


parser = argparse.ArgumentParser(description='Read face metadata from DB and build face embeddings')
parser.add_argument('--config', default='config.yaml', help='Config file path')
args = parser.parse_args()

config = get_config(args.config)
face_data_dir = Path(config.preprocessed_dataset_dir)

session_maker = get_session_maker(config.get_db_url())

with session_maker() as session:
    faces = session.execute(select(Face)).scalars()

    for face in faces:
        img_path = face_data_dir / face.face_image_path
        embedding_obj = DeepFace.represent(
            img_path=img_path,
            model_name=config.embedding_model,
            enforce_detection=False,
            # we already detected faces, so skip detection
            detector_backend='skip',
            # normalization use the same name as the model
            normalization=config.embedding_model
        )

        # embedding is list of D float values where D is embedding size
        # originally, shape is (1, D), dtype is float32
        face.face_embedding = np.array(embedding_obj[0]['embedding'], dtype=np.float32).tobytes()

        session.flush()

    session.commit()
