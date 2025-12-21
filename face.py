"""Real-time face recognition using a webcam.

This script loads a small set of known face images, encodes them using the
`face_recognition` library (dlib), then matches faces from a webcam stream.

Optional: If `deepface` is installed, the script will also display an estimated
age for detected faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import face_recognition

try:
    from deepface import DeepFace  # type: ignore

    _DEEPFACE_AVAILABLE = True
except Exception:
    DeepFace = None  # type: ignore
    _DEEPFACE_AVAILABLE = False


@dataclass(frozen=True)
class KnownPerson:
    name: str
    image_path: Path


DEFAULT_MATCH_THRESHOLD = 0.60


def _encode_face_from_image(image_path: Path) -> Optional["face_recognition.face_encodings"]:
    image = face_recognition.load_image_file(str(image_path))
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return encodings[0]


def load_known_faces(people: List[KnownPerson]) -> Tuple[List, List[str]]:
    encodings: List = []
    names: List[str] = []

    for person in people:
        if not person.image_path.exists():
            raise FileNotFoundError(f"Known face image not found: {person.image_path}")

        encoding = _encode_face_from_image(person.image_path)
        if encoding is None:
            raise ValueError(
                f"No face detected in known image: {person.image_path}. "
                "Use a clear, front-facing photo with a single face."
            )

        encodings.append(encoding)
        names.append(person.name)

    return encodings, names


def estimate_age(face_bgr_image) -> Optional[int]:
    if not _DEEPFACE_AVAILABLE:
        return None

    try:
        result = DeepFace.analyze(face_bgr_image, actions=["age"], enforce_detection=False)
        return int(result[0]["age"])
    except Exception:
        return None


def run_webcam_recognition(
    known_face_encodings: List,
    known_face_names: List[str],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    camera_index: int = 0,
) -> None:
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                raise RuntimeError("Failed to read a frame from the camera.")

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                min_distance_index = distances.argmin()
                min_distance = float(distances[min_distance_index])

                if min_distance < match_threshold:
                    name = known_face_names[min_distance_index]
                    confidence = 1.0 - min_distance
                    confidence_text = f"Confidence: {confidence:.2%}"

                    age_value = estimate_age(frame[top:bottom, left:right])
                    age_text = f"Age: {age_value}" if age_value is not None else ""
                else:
                    name = "Unknown"
                    confidence_text = ""
                    age_text = ""

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    name,
                    (left, max(0, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                if confidence_text:
                    cv2.putText(
                        frame,
                        confidence_text,
                        (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                if age_text:
                    cv2.putText(
                        frame,
                        age_text,
                        (left, bottom + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Replace these with your own images + names for a resume-ready demo.
    people = [
        KnownPerson(name="Asit Kumar", image_path=project_root / "Asit picture.jpg"),
        KnownPerson(name="Dheeraj Kumar", image_path=project_root / "Dheeraj picture.png"),
        KnownPerson(name="Shahid Kapoor", image_path=project_root / "shahid.jpg"),
        KnownPerson(name="Sharaddha Khapra", image_path=project_root / "Sharaddha.jpg"),
        KnownPerson(name="Tamanna Bhatia", image_path=project_root / "Tamanna.jpeg"),
        KnownPerson(name="Virat Kohli", image_path=project_root / "virat.png"),
    ]

    known_face_encodings, known_face_names = load_known_faces(people)
    run_webcam_recognition(known_face_encodings, known_face_names)


if __name__ == "__main__":
    main()