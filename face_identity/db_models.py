import enum

from sqlalchemy import create_engine, String, LargeBinary, Float, Integer, ForeignKey, Enum
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, relationship



class DatasetSplit(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = 'images'

    id = mapped_column(Integer, primary_key=True)
    image_path = mapped_column(String, nullable=False,
                               doc='Relative path of image relative to dataset root directory')
    # TODO: need to make it indexed
    celeb_id = mapped_column(Integer, nullable=False)
    split = mapped_column(Enum(DatasetSplit), nullable=False,
                          doc='Dataset split: train, val, test')

    faces = relationship('Face', back_populates="image")


class Face(Base):
    __tablename__ = 'faces'

    id = mapped_column(Integer, primary_key=True)
    image_id = mapped_column(ForeignKey("images.id"))
    face_image_path = mapped_column(String, nullable=False,
                                    doc='Relative path of face image relative to dataset root directory')
    face_embedding = mapped_column(LargeBinary, nullable=True)
    facial_area = mapped_column(String, nullable=False,
                                doc='JSON string of x, y, w, h, left_eye and right_eye')
    confidence = mapped_column(Float, nullable=False)

    image = relationship("Image", back_populates="faces")


engine = None


def get_session_maker(db_url) -> sessionmaker:
    global engine
    if engine is None:
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        print('Database tables created')

    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
