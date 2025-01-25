import argparse
import json
from pathlib import Path

from deepface import DeepFace
from PIL import Image

from db_models import DatasetSplit, Image as ImageModel, Face, get_session_maker


BACKEND = 'yunet'

parser = argparse.ArgumentParser(description='Detect faces in an image, then save to file and DB')
parser.add_argument('--dataset-dir', required=True, help='Dataest root directory path')
parser.add_argument('--preprocess-dataset-dir', default='preprocessed-dataset',
                    help='Preprocessed data directory path')
args = parser.parse_args()

session_maker = get_session_maker('sqlite:///face_identity.db')

root_data_dir = Path(args.dataset_dir)
assert root_data_dir.exists()

preprocessed_data_dir = Path(args.preprocess_dataset_dir)
preprocessed_data_dir.mkdir(exist_ok=True)

for image_path in root_data_dir.glob('**/*.jpg'):
    celeb_id = image_path.parent.name
    split = DatasetSplit(image_path.parent.parent.name)

    preprocess_image_dir = preprocessed_data_dir / split.value / celeb_id
    preprocess_image_dir.mkdir(parents=True, exist_ok=True)

    img_obj = ImageModel(image_path=image_path.as_posix(), celeb_id=celeb_id, split=split)
    with session_maker() as session:
        session.add(img_obj)
        session.commit()
        session.flush()

        try:
            detected_faces = DeepFace.extract_faces(image_path,
                                                    align=True,
                                                    detector_backend=BACKEND,
                                                    normalize_face=False)
        except ValueError:
            print(f'No face detected in {image_path}')
            continue

        face_objs = []
        for i, detected_face in enumerate(detected_faces):
            face_image_path = preprocess_image_dir / f'{image_path.stem}_{i}{image_path.suffix}'
            face_img = Image.fromarray(detected_face['face'])
            face_img.save(face_image_path.as_posix())

            face_obj = Face(image_id=img_obj.id,
                            face_image_path=face_image_path.as_posix(),
                            facial_area=json.dumps(detected_face['facial_area']),
                            confidence=float(detected_face['confidence']))
            face_objs.append(face_obj)

        session.add_all(face_objs)
        session.commit()
        session.flush()
