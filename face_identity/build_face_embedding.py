import numpy as np
from deepface import DeepFace
from sqlalchemy import select

from db_models import DatasetSplit, Image as ImageModel, Face, get_session_maker



session_maker = get_session_maker('sqlite:///face_identity.db')

# input size is 160x160, embedding size is 128
FACE_MODEL = 'Facenet'

with session_maker() as session:
    faces = session.execute(select(Face)).scalars()

    for face in faces:
        img_path = face.face_image_path
        embedding_obj = DeepFace.represent(
            img_path=img_path,
            model_name=FACE_MODEL,
            enforce_detection=False,
            detector_backend='skip',
            normalization='Facenet'
        )

        # embedding is list of 128 float values
        # originally, shape is (1, 128), dtype is float32
        face.face_embedding = np.array(embedding_obj[0]['embedding'], dtype=np.float32).tobytes()

        session.flush()

    session.commit()
