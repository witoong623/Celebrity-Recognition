import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from db_models import Image, Face, DatasetSplit, get_session_maker


class FaceRecognitionModel:
    embedding_dim = 128

    def __init__(self, db_url):
        self._session_maker = get_session_maker(db_url)

        self._embedding_index_mapper = []

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
                self._embedding_index_mapper.extend([image.celeb_id] * len(image.faces))

        embedding_np = np.array(embedding_list)
        expected_shape = (len(self._embedding_index_mapper), self.embedding_dim)
        assert embedding_np.shape == expected_shape, \
            f'Embedding shape: {embedding_np.shape} not equal to expected shape: {expected_shape}'

        self._index = faiss.IndexFlatL2(self.embedding_dim)
        self._index.add(embedding_np)
        print(f'Finished building {len(self._embedding_index_mapper)} embeddings, index size: {self._index.ntotal}')

    def recognize(self, query_embeddings: np.ndarray) -> tuple[list[float], list[int]]:
        '''
        Recognize faces by finding nearest neighbor embeddings
        Args:
            query_embeddings: numpy array of shape (n, d) where n is number of queries
                            and d is embedding dimension
        Returns:
            distances: distances to nearest neighbors
            indices: original indices of nearest neighbors
        '''
        # Validate input shape
        if len(query_embeddings.shape) != 2:
            raise ValueError('Query embeddings must be 2D array')

        # Search top 1 nearest neighbor
        distances, indices = self._index.search(query_embeddings, k=1)

        # Flatten results since k=1
        distances = distances.flatten()
        indices = indices.flatten()

        # Map indices back to original indices
        mapped_indices = [self._embedding_index_mapper[idx] for idx in indices]

        return distances, mapped_indices

