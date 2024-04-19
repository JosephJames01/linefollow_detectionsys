import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import glob

class ObjectDetection:
    def __init__(self):
        self.device = 'cpu'
        self.model = self.load_model()
        self.output_position = 1  # Initialize the output position counter

    def load_model(self):
        model = YOLO('models/best(3).pt')  # Load the YOLOv8 model
        model.fuse()
        return model

    def predict(self, image_path):
        results = self.model(image_path)
        return results

    def plot_bboxes(self, results, frame, class_name_to_detect='mug'):
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            class_ids = result.boxes.cls.numpy()
            class_names = result.names

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                confidence = confidences[i]
                class_id = class_ids[i]
                detected_class_name = class_names[class_id]

                if detected_class_name == class_name_to_detect and confidence > 0.85:  # Threshold can be adjusted
                    color = (0, 255, 0)  # Green color for the bounding box
                    text_color = (255, 255, 255)  # White color for the text

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{detected_class_name}: {confidence:.2f}"

                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        return frame

    def wait_for_new_image(self, directory_path):
        """
        Waits for a new image to appear in the specified directory and returns its path.
        """
        while True:
            files = os.listdir(directory_path)
            if files:
                return os.path.join(directory_path, files[0])
            else:
                time.sleep(1)  # Wait for 1 second before checking again

    def delete_files_in_directory(self, directory_path):
        """
        Deletes all files in the specified directory.
        """
        files = glob.glob(f'{directory_path}/*')
        for f in files:
            os.remove(f)

    def capture(self):
        directory_path = 'results'  # Directory to monitor for new images
        
        while True:
            image_path = self.wait_for_new_image(directory_path)
            if image_path:
                image = cv2.imread(image_path)
                if image is not None:
                    results = self.predict(image_path)
                    frame = image
                    frame_with_boxes = self.plot_bboxes(results, frame)

                    mugs_count = 0
                    for result in results:
                        class_name = result.names[result.boxes.cls.numpy().argmax()]
                        confidence = result.boxes.conf.numpy().max()
                        if class_name == 'mug' and confidence > 0.85:
                            mugs_count += 1
                            print(f'position: {self.output_position}, mugs: {mugs_count}, confidence: {confidence:.2f}')

                    output_filename = os.path.basename(image_path)  # Use the same filename for the output image
                    cv2.imwrite(f'positions/{output_filename}', frame_with_boxes)  # Save the image with bounding boxes
                    self.output_position += 1  # Increment the output position for the next image
                    
                    # Optionally, delete the processed image to wait for a new one
                    os.remove(image_path)

        # After exiting the loop, delete all files in the directory
        self.delete_files_in_directory(directory_path)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    object_detection = ObjectDetection()
    object_detection.capture()
