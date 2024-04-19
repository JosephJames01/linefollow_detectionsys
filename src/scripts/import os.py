import os
import cv2
while True:
 key = cv2.waitKeyEx(0) 
 if key == ord('x'):
# Define the file path and name
  file_path = "hello_you.txt"

# Check if the file already exists
  if os.path.exists(file_path):
    print(f"File '{file_path}' already exists.")
  else:
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the content to the file
        file.write("Hello you")

    print(f"File '{file_path}' created and written to.")