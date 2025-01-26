import argparse
import json
from pathlib import Path

from deepface import DeepFace
from PIL import Image

from db_models import DatasetSplit, Image as ImageModel, Face, get_session_maker
from config import get_config


parser = argparse.ArgumentParser(description='Detect faces in an image, then save to file and DB')
parser.add_argument('--config', default='config.yaml', help='Config file path')
args = parser.parse_args()

config = get_config(args.config)

session_maker = get_session_maker(config.get_db_url())

root_data_dir = Path(config.dataset_dir)
assert root_data_dir.exists()

preprocessed_data_dir = Path(config.preprocessed_dataset_dir)
preprocessed_data_dir.mkdir(exist_ok=True)

no_face_image_count = 0

for image_path in root_data_dir.glob('**/*.jpg'):
    celeb_id = image_path.parent.name
    split = DatasetSplit(image_path.parent.parent.name)

    preprocess_image_dir = preprocessed_data_dir / split.value / celeb_id
    preprocess_image_dir.mkdir(parents=True, exist_ok=True)

    with session_maker() as session:
        img_obj = ImageModel(image_path=image_path.relative_to(root_data_dir).as_posix(),
                             celeb_id=celeb_id, split=split)
        session.add(img_obj)
        session.flush()

        try:
            detected_faces = DeepFace.extract_faces(image_path,
                                                    align=True,
                                                    detector_backend=config.face_detector_backend,
                                                    normalize_face=False)
        except ValueError:
            print(f'No face detected in {image_path}')
            no_face_image_count += 1
            continue

        face_objs = []
        for i, detected_face in enumerate(detected_faces):
            face_image_path = preprocess_image_dir / f'{image_path.stem}_{i}{image_path.suffix}'
            face_img = Image.fromarray(detected_face['face'])
            face_img.save(face_image_path.as_posix())

            face_obj = Face(image_id=img_obj.id,
                            face_image_path=face_image_path.relative_to(preprocessed_data_dir).as_posix(),
                            facial_area=json.dumps(detected_face['facial_area']),
                            confidence=float(detected_face['confidence']))
            face_objs.append(face_obj)

        session.add_all(face_objs)
        session.flush()
        session.commit()

print('Finished detecting face in dataset.\n'
      f'No face detected in {no_face_image_count} images')
