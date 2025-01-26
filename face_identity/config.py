import yaml
from pydantic import BaseModel


class Config(BaseModel):
    # SQLite database filename end with .db
    db_file: str

    # root dataset directory. Inside contains train and test directory.
    dataset_dir: str
    # root of preprocessed dataset directory.
    preprocessed_dataset_dir: str

    face_detector_backend: str
    embedding_model: str
    # target L2 threshold to determine face similarity
    L2_threshold: float

    def get_db_url(self):
        ''' Return SQLite database URL '''
        return f'sqlite:///{self.db_file}'


def get_config(config_filepath) -> Config:
    with open(config_filepath, 'r') as f:
        return Config(**yaml.safe_load(f))
