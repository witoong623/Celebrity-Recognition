import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from db_models import Image, Face, DatasetSplit, get_session_maker


class FaceRecognitionModel:
    embedding_dim = 128

    def __init__(self, db_url):
        self._session_maker = get_session_maker(db_url)

        self._celeb_id_mapper = []
        self._face_id_mapper = []

        # build embeddings index for each image
        embedding_list = []
        with self._session_maker() as session:
            stmt = (select(Image)
                    .join(Face)
                    .where(Image.split == DatasetSplit.TRAIN, Face.outlier == False)
                    .order_by(Image.id)
                    .options(joinedload(Image.faces)))
            for image in session.execute(stmt).unique().scalars().all():
                embedding_list.extend([np.frombuffer(face.face_embedding, dtype=np.float32) for face in image.faces])
                self._celeb_id_mapper.extend([image.celeb_id] * len(image.faces))
                self._face_id_mapper.extend([face.id for face in image.faces])

        embedding_np = np.array(embedding_list)
        expected_shape = (len(self._celeb_id_mapper), self.embedding_dim)
        assert embedding_np.shape == expected_shape, \
            f'Embedding shape: {embedding_np.shape} not equal to expected shape: {expected_shape}'

        self._index = faiss.IndexFlatL2(self.embedding_dim)
        self._index.add(embedding_np)
        print(f'Finished building {len(self._celeb_id_mapper)} embeddings, index size: {self._index.ntotal}')

    def recognize(self, query_embeddings: np.ndarray, return_face_ids=False) -> tuple[list[float], list[int]]:
        '''
        Recognize faces by finding nearest neighbor embeddings
        Args:
            query_embeddings: numpy array of shape (n, d) where n is number of queries
                            and d is embedding dimension
            return_face_ids: whether to return ID of nearest face in database
        Returns:
            distances: distances to nearest neighbors
            celeb_ids: original indices of nearest neighbors
            face_ids: IDs of nearest faces in database
        '''
        # Validate input shape
        if len(query_embeddings.shape) != 2:
            raise ValueError('Query embeddings must be 2D array')

        # Search top 1 nearest neighbor
        distances, indices = self._index.search(query_embeddings, k=1)

        # Flatten results since k=1
        distances = distances.flatten()
        indices = indices.flatten()

        # Map vector indices to celeb ids
        mapped_celeb_ids = [self._celeb_id_mapper[idx] for idx in indices]

        if return_face_ids:
            return distances, mapped_celeb_ids, [self._face_id_mapper[idx] for idx in indices]
        else:
            return distances, mapped_celeb_ids
