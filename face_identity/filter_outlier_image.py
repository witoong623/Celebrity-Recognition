import argparse
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import select

from config import get_config
from db_models import Face, DatasetSplit, Image, get_session_maker


parser = argparse.ArgumentParser(description='Filter out outlier face images using data from DB')
parser.add_argument('--config', default='config.yaml', help='Config file path')
args = parser.parse_args()

config = get_config(args.config)

session_maker = get_session_maker(config.get_db_url())

with session_maker() as session:
    all_celeb_ids = session.execute(select(Image.celeb_id)
                                    .where(Image.split == DatasetSplit.TRAIN)).unique().scalars().all()

with session_maker() as session:
    for celeb_id in all_celeb_ids:
        faces = session.execute(select(Face)
                                .join(Image)
                                .where(Image.celeb_id == celeb_id,
                                       Image.split == DatasetSplit.TRAIN)).scalars().all()

        if len(faces) < 5:
            print(f'Celeb {celeb_id} has less than 5 faces, so it isn\'t effective to cluster. Skip')
            continue

        face_embeddings = [np.frombuffer(face.face_embedding, dtype=np.float32) for face in faces]
        face_embeddings = np.stack(face_embeddings, axis=0)

        face_clusters_indexes = DBSCAN(eps=10, min_samples=1, metric='euclidean').fit_predict(face_embeddings)
        assert len(face_clusters_indexes) == len(faces)

        cluster_counter = Counter(face_clusters_indexes)
        top_2_clusters_count = sum([count for _, count in cluster_counter.most_common(2)])
        # need top 2 clusters to have at least 50% of all faces
        # otherwise, it's likely that the faces are too diverse to be clustered and find outliers
        if top_2_clusters_count / len(faces) < 0.5:
            print(f'Celeb {celeb_id} has too diverse faces to be clustered. Skip')
            continue

        outlider_count = 0
        # -1 is outlider, mark all faces in this cluster as outlier
        if -1 in cluster_counter:
            for item_idx, cluster_idx in enumerate(face_clusters_indexes):
                if cluster_idx == -1:
                    faces[item_idx].outlier = True
                    outlider_count += 1
            del cluster_counter[-1]

        for cluster_idx, member_count in cluster_counter.items():
            # if the cluster has more than 1 member, then it's not an outlier
            # TODO: some celeb has many children's images, so children cluster size is more than 1
            if member_count > 1:
                continue

            for item_idx, c_idx in enumerate(face_clusters_indexes):
                if c_idx == cluster_idx:
                    faces[item_idx].outlier = True
                    outlider_count += 1

        print(f'Celeb {celeb_id} has {outlider_count} outliers')
        session.flush()

    session.commit()
