import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from db_models import get_session_maker, Image, Face, DatasetSplit
from recognition import FaceRecognition
# import ace_tools as tools


session_maker = get_session_maker('sqlite:///face_identity.db')

# get test embedding
test_embeddings = []
ground_truth_labels = []
with session_maker() as session:
    stmt = select(Image).options(joinedload(Image.faces)).where(Image.split == DatasetSplit.TEST).order_by(Image.id)
    for image in session.execute(stmt).unique().scalars().all():
        if len(image.faces) > 1:
            largest_face = max(image.faces, key=lambda face: face.face_area)
        elif len(image.faces) == 1:
            largest_face = image.faces[0]
        else:
            # this image doesn't have face. It means that our detector didn't detect it
            # Since we are going to use to same face detector anyway, we will treat it as true negative
            # so we will skip this image
            continue

        test_embeddings.append(np.frombuffer(largest_face.face_embedding, dtype=np.float32))
        ground_truth_labels.append(image.celeb_id)

test_embeddings = np.array(test_embeddings)

face_recognizer = FaceRecognition('sqlite:///face_identity.db')
predicted_distances, predicted_labels = face_recognizer.recognize(test_embeddings)

# Set threshold for Euclidean Distance
threshold = 80

predicted_labels = [label if distance < threshold else -1 for distance, label in zip(predicted_distances, predicted_labels)]

# Convert labels to numpy arrays
predicted_labels = np.array(predicted_labels, dtype=np.int64)
ground_truth_labels = np.array(ground_truth_labels, dtype=np.int64)

# Compute TP, FP, FN based on actual & predicted labels
TP = np.sum((predicted_labels != -1) & (predicted_labels == ground_truth_labels))  # Correctly identified celebrities
FP = np.sum((predicted_labels != -1) & (predicted_labels != ground_truth_labels))  # Wrong celebrity assigned
FN = np.sum((predicted_labels == -1) & (ground_truth_labels != -1))  # Missed celebrity ("Can't specify")
TN = np.sum((predicted_labels == -1) & (ground_truth_labels == -1))  # Correctly rejected non-celebrities

# Compute Precision, Recall, and F1-score
precision = precision_score(ground_truth_labels[ground_truth_labels != -1], predicted_labels[ground_truth_labels != -1], average='macro', zero_division=0)
recall = recall_score(ground_truth_labels[ground_truth_labels != -1], predicted_labels[ground_truth_labels != -1], average='macro', zero_division=0)
f1 = f1_score(ground_truth_labels[ground_truth_labels != -1], predicted_labels[ground_truth_labels != -1], average='macro', zero_division=0)

# Display results
metrics_results = {
    "Threshold": threshold,
    "True Positives": TP,
    "False Positives": FP,
    "False Negatives": FN,
    "True Negatives": TN,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
}

# Show metrics in a DataFrame
import pandas as pd
df_metrics = pd.DataFrame([metrics_results])
print(df_metrics)
# tools.display_dataframe_to_user(name="Face Recognition Evaluation Metrics", dataframe=df_metrics)