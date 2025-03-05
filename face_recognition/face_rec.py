import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from database.milvus_db import face_collection
from dotenv import load_dotenv

load_dotenv()

# Check for CUDA availability and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Initialize FaceNet model, ensuring it's moved to the appropriate device
try:
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("FaceNet model loaded successfully.")
except Exception as e:
    print(f"Failed to load FaceNet model: {e}")
    facenet_model = None

# ----------------------------------------------------------------------
# 3. Face Detection (OpenCV)
# ----------------------------------------------------------------------

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----------------------------------------------------------------------
# 4. Face Registration
# ----------------------------------------------------------------------

def register_patient_face(patient_id, face_image_path):
    """
    Registers a patient's face by detecting the face, extracting its embedding,
    and storing the embedding in Milvus.
    """
    if facenet_model is None or face_collection is None:
        print("FaceNet model or Milvus collection not initialized.")
        return False

    img = cv2.imread(face_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected")
        return False

    (x, y, w, h) = faces[0]
    face_roi = img[y:y + h, x:x + w]

    # Resize the face ROI to a fixed size for FaceNet
    face_roi = cv2.resize(face_roi, (160, 160))

    # Convert the OpenCV image (BGR) to PIL format (RGB) for FaceNet
    face_image_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

    # Ensure the image is processed on the correct device
    try:
        with torch.no_grad():
            face_embedding = facenet_model(torch.stack([torch.tensor(np.array(face_image_pil)).permute(2, 0, 1).float().to(device) / 255])).cpu().numpy().tolist()[0]

            # Normalize the embedding vector
            face_embedding = face_embedding / np.linalg.norm(face_embedding)  #Normalize the vector
            print(f"Face embedding during registration: {face_embedding}")  # Add this line

        data = [
            [patient_id],
            [face_embedding]
        ]
        try:
            face_collection.insert(data)
            face_collection.flush()
            print(f"Face registered for patient {patient_id}")
            return True
        except Exception as e:
            print(f"Failed to insert data into Milvus: {e}")
            return False
    except Exception as e:
        print(f"Error during FaceNet processing: {e}")
        return False

# ----------------------------------------------------------------------
# 5. Face Recognition
# ----------------------------------------------------------------------

def scan_face_and_get_patient_id(face_image_path):
    """
    Scans a face image, extracts its embedding using FaceNet, and compares it with the embeddings
    stored in Milvus to identify the patient.
    """
    if facenet_model is None or face_collection is None:
        print("FaceNet model or Milvus collection not initialized.")
        return None

    img = cv2.imread(face_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected")
        return None

    (x, y, w, h) = faces[0]
    face_roi = img[y:y + h, x:x + w]

    # Resize the face ROI to a fixed size for FaceNet
    face_roi = cv2.resize(face_roi, (160, 160))

    # Convert the OpenCV image (BGR) to PIL format (RGB) for FaceNet
    face_image_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

    try:
        with torch.no_grad():
            new_embedding = facenet_model(torch.stack([torch.tensor(np.array(face_image_pil)).permute(2, 0, 1).float().to(device) / 255])).cpu().numpy().tolist()

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}  # Adjust ef for performance
        }
        try:
            results = face_collection.search(
                data=new_embedding,
                anns_field="embedding",
                param=search_params,
                limit=1,
                output_fields=["patient_id"]
            )

            if results and results[0]:
                best_match_id = results[0][0].entity.get("patient_id")
                similarity_score = results[0][0].distance #The lower the cosine distance the better

                print(f"Similarity score: {similarity_score}")  # Add this line

                # Save the face ROI to a file
                cv2.imwrite("recognized_face.jpg", face_roi)  # Add this line

                if  similarity_score > 0.3:  # Adjust threshold as needed (cosine similarity ranges from 0 to 2, 0 being a perfect match and 2 being completely different )
                    print(f"Face recognized as {best_match_id} with similarity {similarity_score}")
                    return best_match_id
                else:
                    print("Face not recognized.")
                    return None
            else:
                print("No search results.")
                return None

        except Exception as e:
            print(f"Failed to search Milvus: {e}")
            return None

    except Exception as e:
        print(f"Error during FaceNet processing: {e}")
        return None

# ----------------------------------------------------------------------
# 6. Camera Integration (Example using OpenCV) - adapted for FaceNet and to close after face is identified
# ----------------------------------------------------------------------

def recognize_face_from_camera():
    """
    Captures a frame from the camera, detects faces, extracts features using FaceNet,
    and attempts to identify the patient using Milvus. Closes after identifying a face.
    """
    if facenet_model is None or face_collection is None:
        print("FaceNet model or Milvus collection not initialized.")
        return None

    video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera
    recognized_patient_id = None  # Store the recognized patient ID

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_roi = frame[y:y + h, x:x + w]

            # Resize the face ROI to a fixed size for FaceNet
            face_roi = cv2.resize(face_roi, (160, 160))

            # Convert the OpenCV image (BGR) to PIL format (RGB) for FaceNet
            face_image_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

            try:
                with torch.no_grad():
                    new_embedding = facenet_model(torch.stack([torch.tensor(np.array(face_image_pil)).permute(2, 0, 1).float().to(device) / 255])).cpu().numpy().tolist()

                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": 64}  # Adjust ef for performance
                }
                results = face_collection.search(
                    data=new_embedding,
                    anns_field="embedding",
                    param=search_params,
                    limit=1,
                    output_fields=["patient_id"]
                )

                if results and results[0]:
                    best_match_id = results[0][0].entity.get("patient_id")
                    similarity_score = results[0][0].distance #The lower the cosine distance the better

                    if (1 - similarity_score) > 0.7:  # Adjust threshold as needed
                        text = f"Recognized: {best_match_id}"
                        recognized_patient_id = best_match_id  # Store the recognized ID
                    else:
                        text = "Face not recognized."
                else:
                    text = "Face not recognized."

                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")
                text = "Error"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Close after identifying a face
        if recognized_patient_id:
            break

    # When everything's done, release the capture and destroy windows
    video_capture.release()
    cv2.destroyAllWindows()
    return recognized_patient_id  # Return the recognized ID

# ----------------------------------------------------------------------
# 7. Integrating into the main LangGraph flow (Adapt as needed)
# ----------------------------------------------------------------------

def main():
    # Example usage:
    face_image_path = "patient_image.jpg"  # Replace with actual path

    # **Register the test patient's face:**
    register_patient_face("test_patient", face_image_path)  # Uncomment to register a new face

    # Identify face from camera
    recognized_patient_id = recognize_face_from_camera()  # Capture the recognized ID

    if recognized_patient_id:
      print(f"Recognized Patient ID: {recognized_patient_id}")
    else:
      print("No face has been recognized")

if __name__ == "__main__":
    main()