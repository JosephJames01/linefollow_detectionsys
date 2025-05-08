import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import glob


class ObjectDetection:
   # build the object 
    def __init__(self):
        self.device = 'cpu'
        self.model = self.load_model()
        self.output_position = 1  # Initialize the output position counter
        self.confidence_scores = {}
        self.total_position = 0
        self.total_mugs = 0 
    # load the model 
    def load_model(self):
        model = YOLO('models/best(3).pt')  # Load the YOLOv8 model
        model.fuse()
        return model
   #predict 
    def predict(self, image_path):
        results = self.model(image_path)
        return results
     #plot the bounding boxes around the detected class
    def plot_bboxes(self, results, frame, class_name_to_detect='mug'):
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            class_ids = result.boxes.cls.numpy()
            class_names = result.names
  
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]   # box coordinates
                confidence = confidences[i]                      # confidences
                class_id = class_ids[i]                          # class id
                detected_class_name = class_names[class_id]

                if detected_class_name == class_name_to_detect and confidence > 0.85:  # Threshold can be adjusted, 0.85 for high accuracy 
                    color = (0, 0, 255)  # Red for bounding box 
                    text_color = (255, 255, 255)  # White color for the text

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{detected_class_name}: {confidence:.2f}"

                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)
                    cv2.rectangle(frame, (x1, y2 - text_height - 4), (x1 + text_width, y2), color, -1) #draw rectangle 
                    cv2.putText(frame, text, (x1, y2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 3) #draw text 
                    #noting output position 
                    position = self.output_position   
                     
                    self.confidence_scores[position] = confidence
        return frame
   # wait for image 
    def wait_for_new_image(self, directory_path):
        """
        Waits for a new image to appear in the specified directory and returns its path.
        """
        while True:
            for i in range(1, 1000):  # Check up to 1000 possible image files
                image_path = os.path.join(directory_path, f'outputposition{i}.jpg')
                if os.path.isfile(image_path):
                    return image_path
                
    
            time.sleep(1)  # Wait for 1 second before checking again
            print("Waiting for image")

   
    def delete_files_in_directory(self, directory_path):
        """
        Deletes all files in the specified directory.
        """
        files = glob.glob(f'{directory_path}/*')
        for f in files:
            os.remove(f)
    # capture the image 
    def capture(self):
     directory_path = 'results'  # Directory to monitor for new images
    
     while True:
       # wait for image in directory path 
        image_path = self.wait_for_new_image(directory_path)
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                results = self.predict(image_path)
                frame = image
                 #increase total possitions
                self.total_position = self.total_position + 1
                mugs_count = 0   #initialize mug count 
                frame_with_boxes = frame  # Initialize with the original frame
                for result in results:
                # Check if any detections were made
                 if len(result.boxes) > 0:
                    for result in results:
                        class_name = result.names[result.boxes.cls.numpy().argmax()]
                        confidence = result.boxes.conf.numpy().max()
                          # printing terminal information for tracking purposes
                        if class_name == 'mug' and confidence > 0.85: 
                            mugs_count += 1
                            self.total_mugs = self.total_mugs + 1
                            frame_with_boxes = self.plot_bboxes(results, frame)
                              
                            print(f'Image received: position: {self.output_position}, mugs: {mugs_count}, confidence: {confidence:.2f}, class: {class_name}, total postions: {self.total_position}, total mugs:{self.total_mugs}')
                        else: 
                            print(f'Image received but no or likely no mug. position: {self.output_position}, mugs: {mugs_count}, confidence: {confidence:.2f}, class: {class_name}, total postions: {self.total_position}, total mugs:{self.total_mugs}')
                 else:
                    print(f'No detections. position: {self.output_position}, total postions: {self.total_position}, total mugs:{self.total_mugs}')
                
                output_filename = os.path.basename(image_path)  # Use the same filename for the output image
                cv2.imwrite(f'positions/{output_filename}', frame_with_boxes)  # Save the image with bounding boxes
                self.output_position += 1  # Increment the output position for the next image
                
                # Delete the processed image to wait for a new one
                os.remove(image_path)

                    
                 
              

                # Calculate the average confidence score
                
                

        # After exiting the loop, delete all files in the directory
        

        cv2.destroyAllWindows()
# main method
if __name__ == '__main__':
    object_detection = ObjectDetection()
    object_detection.capture()
    
