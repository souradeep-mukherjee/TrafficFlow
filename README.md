# Automated Traffic Control System 

Overview : Automated Traffic Control System using YOLOv8 , Python , Computer Vision 
This project implements a real - time intelligent traffic control system that uses deep learning and image processing to optimize traffic signal decisions dynamically . 

![Diagrammatic representation  (1)](https://github.com/souradeep-mukherjee/TrafficFlow/blob/master/flow.png)

Key features : 
1.  Object detection : Vehicles , Pedestrians and Emergency Vehicles 
2. Priority Calculation Algorithm 
3.  Resolves Starvation Problem 
4. Highest priority given to the emergency vehicles 
5.  Custom Simulation 

Technologies used : 
1. YOLOv8 (via Ultralytics) for real time object detection .
2. OpenCV for image handling .
3. Python for decision making logic and data processing .
4. NumPy for numerical operations . 
6. Tkinter for creating the GUI .

Installations required : 
Here are the **installation requirements** for running the **Automated Traffic Control System** project:

---

### **1. Required Python Packages**
Install the following packages using `pip`:

pip install opencv-python numpy pillow ultralytics tk


#### **Breakdown of Packages:**
| Package | Purpose |
|---------|---------|
| `opencv-python` (cv2) | Computer vision (video/image processing, object detection) |
| `numpy` | Numerical computations (array operations, image handling) |
| `pillow` (PIL) | Image processing for Tkinter GUI |
| `ultralytics` (YOLOv8) | Deep learning-based object detection (vehicles, pedestrians) |
| `tk` (Tkinter) | GUI framework (pre-installed with Python) |

---

### **2. Optional Packages (For Advanced Features)**
These are included in the project but may require additional setup:

pip install matplotlib pandas scipy

- **`matplotlib`**: Used for data visualization (if generating graphs).
- **`pandas`**: For structured data analysis (if exporting reports).
- **`scipy`**: Advanced mathematical computations (if needed for simulations).



### **3. Additional Setup (If Using GPU Acceleration)**
For **faster YOLOv8 detection**, install **PyTorch with CUDA** (if you have an NVIDIA GPU):

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

*(Replace `cu117` with your CUDA version if different.)*



### **4. Download YOLOv8 Weights (Automatically Handled)**
The project uses **YOLOv8n (nano model)** for object detection. The weights (`yolov8n.pt`) are automatically downloaded when you first run the code:

from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Auto-downloads if not found


### **5. Verify Installation**
Run this Python snippet to check if all dependencies are installed:

import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from ultralytics import YOLO

print("All dependencies installed successfully!")

### **Troubleshooting**
1. **If OpenCV fails to install**, try:
   
   pip install opencv-python-headless
   
3. **If Tkinter is missing** (common on Linux):

   sudo apt-get install python3-tk  # Debian/Ubuntu
   sudo dnf install python3-tkinter # Fedora
   
4. **If YOLO fails to download weights**, manually download `yolov8n.pt` from [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) and place it in the project directory.

### **Final Notes**
- **OS Compatibility**: Works on **Windows, macOS, and Linux**.
- **Python Version**: Requires **Python 3.7+**.
- **Hardware**: Runs on CPU by default, but GPU (CUDA) is recommended for real-time video processing.

After installation, simply run:

python ATMS02.py

to launch the **Traffic Control System GUI**. 


How it works : 
1. Upload Traffic Images from local device (min. 2 or max. 4) required to depict 2-4 lanes of a road.
OR
Upload Traffic Video. 
Custom simulate the number of lanes , vehicles , pedestrians and emergency vehicles .
OR 
Upload a traffic dataset.  

3. Object Detection using YOLOv8 : Identifies and counts vehicles, pedestrians, and emergency vehicles in each lane.

4. Priority Calculation Logic  :  For each lane, a priority score is calculated based on - 
   - Vehicle count (weight: 1.0 per vehicle)
   - Pedestrian count (weight: 0.5 per pedestrian)
   - Emergency vehicles (absolute priority: 100 points each)
   - Starvation factor: Lanes waiting too long receive a boost (0.2 points per second)
   - Red light duration penalty: Additional priority for lanes stuck at red light

3. Decision: The lane with the highest priority score receives the green light.

4. Fairness Control: To prevent starvation, lanes that don't get green light accumulate additional priority over time.

5. Emergency Override: Emergency vehicles automatically get highest priority regardless of other factors.

   
![Screenshot 2025-04-23 124146](https://github.com/user-attachments/assets/04a86697-3cfc-467f-bc50-8ea491385985)
![Screenshot 2025-04-23 124154](https://github.com/user-attachments/assets/30d06ace-2266-4c14-8ce3-0e7cab20a5d8)
![Screenshot (78)](https://github.com/user-attachments/assets/577c795f-50bb-4225-8cbd-cb6b8ff0b2be)
![Screenshot (79)](https://github.com/user-attachments/assets/8b1de02d-0c61-4c79-b4b0-6514550b57a7)!
![Screenshot (77)](https://github.com/user-attachments/assets/b04f2f37-63de-4565-baf8-4e1046d0745e)
