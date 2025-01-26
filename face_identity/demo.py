import argparse

from config import get_config
from embedding import get_face_embedding
from recognition_models import FaceRecognitionModel

parser = argparse.ArgumentParser(description='Demo script for celeb face recognition')
parser.add_argument('--image', required=True, help='Image file path')
parser.add_argument('--config', default='config.yaml', help='Config file path')
parser.add_argument('--threshold', type=float, default=None,
                    help='L2 distance threshold for face recognition, valid value between 0 and infinity')
args = parser.parse_args()

config = get_config(args.config)
recognizer = FaceRecognitionModel(config.get_db_url())

query_embedding = get_face_embedding(args.image, config)

distance, predict_celeb_id = recognizer.recognize(query_embedding)

if args.threshold is not None and distance > args.threshold:
    print(f'No matching face found, distance: {distance}, threshold: {args.threshold}')

print(f'Predicted celeb ID: {predict_celeb_id[0]}, distance: {distance[0]}')
