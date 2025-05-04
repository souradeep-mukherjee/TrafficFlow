import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import types
from collections import defaultdict, deque
import random
import os
import threading
from datetime import datetime


class TrafficSimulator:
    def __init__(self):
        self.background = self.create_road_background()
        self.vehicle_templates = self._load_vehicle_templates()
        self.pedestrian_template = self._load_pedestrian_template()
        self.traffic_light_img = self._create_traffic_light_template()
        
    def create_road_background(self, width=800, height=600):
        """Create road background with lanes"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:,:] = (50, 50, 50)  # Road color
        
        # Lane markings
        cv2.line(img, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        for y in range(0, height, 30):
            cv2.line(img, (width//4, y), (width//4, y+15), (255, 255, 255), 2)
            cv2.line(img, (3*width//4, y), (3*width//4, y+15), (255, 255, 255), 2)
        return img
    
    def _load_vehicle_templates(self):
        """Create and return vehicle templates"""
        return {
            'car': self._create_car_template(),
            'bus': self._create_bus_template(),
            'truck': self._create_truck_template(),
            'motorcycle': self._create_motorcycle_template(),
            'ambulance': self._create_ambulance_template()
        }
        
    def _create_traffic_light_template(self):
        """Create a traffic light image"""
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (50, 100), (30, 30, 30), -1)  # Pole
        cv2.circle(img, (25, 25), 20, (0, 0, 0), -1)  # Red light (off)
        cv2.circle(img, (25, 50), 20, (0, 0, 0), -1)  # Yellow light (off)
        cv2.circle(img, (25, 75), 20, (0, 0, 0), -1)  # Green light (off)
        return img
    
    def _load_pedestrian_template(self):
        """Create and return pedestrian template"""
        return self._create_pedestrian_template()
    
    def _create_car_template(self):
        template = np.zeros((60, 100, 3), dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (90, 50), (0, 0, 255), -1)
        return template
    
    def _create_bus_template(self):
        template = np.zeros((80, 120, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (120, 80), (255, 0, 0), -1)
        for x in [10, 40, 70]:
            cv2.rectangle(template, (x, 10), (x+20, 30), (255, 255, 255), -1)
        return template
    
    def _create_truck_template(self):
        template = np.zeros((70, 150, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 30), (150, 70), (0, 255, 0), -1)
        cv2.rectangle(template, (0, 0), (50, 30), (0, 255, 0), -1)
        return template
    
    def _create_motorcycle_template(self):
        template = np.zeros((40, 60, 3), dtype=np.uint8)
        cv2.circle(template, (15, 35), 10, (255, 255, 0), -1)
        cv2.circle(template, (45, 35), 10, (255, 255, 0), -1)
        cv2.line(template, (15, 25), (45, 15), (255, 255, 0), 3)
        return template
    
    def _create_ambulance_template(self):
        template = np.zeros((70, 120, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (120, 70), (255, 0, 0), -1)
        cv2.line(template, (0, 0), (60, 35), (255, 255, 255), 3)
        cv2.line(template, (60, 0), (0, 35), (255, 255, 255), 3)
        return template
    
    def _create_pedestrian_template(self):
        template = np.zeros((80, 40, 3), dtype=np.uint8)
        # Head
        cv2.circle(template, (20, 10), 8, (200, 150, 150), -1)
        # Body
        cv2.line(template, (20, 18), (20, 50), (255, 0, 0), 2)
        # Arms
        cv2.line(template, (20, 25), (5, 35), (255, 0, 0), 2)
        cv2.line(template, (20, 25), (35, 35), (255, 0, 0), 2)
        # Legs
        cv2.line(template, (20, 50), (10, 70), (255, 0, 0), 2)
        cv2.line(template, (20, 50), (30, 70), (255, 0, 0), 2)
        return template
    
    def overlay_template(self, background, template, x, y):
        """Overlay template image onto background"""
        h, w = template.shape[:2]
        # Ensure we don't try to overlay outside the image boundaries
        if y+h > background.shape[0] or x+w > background.shape[1]:
            h = min(h, background.shape[0] - y)
            w = min(w, background.shape[1] - x)
            if h <= 0 or w <= 0:
                return  # Nothing to overlay
            
        for c in range(3):
            background[y:y+h, x:x+w, c] = np.where(
                template[:h, :w, c] > 0,
                template[:h, :w, c],
                background[y:y+h, x:x+w, c])
            
    def update_traffic_light(self, img, is_green):
        """Update traffic light status on image"""
        light_img = self.traffic_light_img.copy()
        if is_green:
            cv2.circle(light_img, (25, 75), 20, (0, 255, 0), -1)  # Green on
            cv2.circle(light_img, (25, 25), 20, (0, 0, 0), -1)     # Red off
        else:
            cv2.circle(light_img, (25, 25), 20, (0, 0, 255), -1)  # Red on
            cv2.circle(light_img, (25, 75), 20, (0, 0, 0), -1)    # Green off
        
        # Overlay traffic light
        x = img.shape[1] - light_img.shape[1] - 20
        y = 20
        self.overlay_template(img, light_img, x, y)
    
    def generate_lane_image(self, vehicles, pedestrians=0, emergency=False):
        """Generate simulated traffic image with vehicles and pedestrians"""
        img = self.background.copy()
        height, width = img.shape[:2]
        
        # Place vehicles
        for vehicle_type in vehicles:
            template = self.vehicle_templates.get(vehicle_type.lower(), self.vehicle_templates['car'])
            
            # Ensure valid placement coordinates
            min_x = width//4 + 20
            max_x = width//2 - 20 - template.shape[1]
            if max_x > min_x:
                x = random.randint(min_x, max_x)
            else:
                x = min_x
                
            min_y = 20
            max_y = height - 20 - template.shape[0]
            y = random.randint(min_y, max_y)
            
            self.overlay_template(img, template, x, y)
            
            if emergency and vehicle_type.lower() == "ambulance":
                cv2.rectangle(img, (x, y-10), (x+template.shape[1], y-5), (0, 255, 255), -1)
        
        # Place pedestrians
        for _ in range(pedestrians):
            template = self.pedestrian_template
            min_x = width//2 + 20
            max_x = width - 20 - template.shape[1]
            x = random.randint(min_x, max_x)
            y = random.randint(20, height - 20 - template.shape[0])
            self.overlay_template(img, template, x, y)
        
        return img


class TrafficLane:
    def __init__(self, name, roi_points):
        self.name = name
        self.roi_points = np.array(roi_points, np.int32)
        self.vehicle_count = 0
        self.congestion_level = 0
        self.is_green = False
        self.green_time = 0
        self.history = deque(maxlen=10)  # Store recent congestion levels
    
    def calculate_congestion(self, vehicle_count):
        """Update congestion level based on vehicle count"""
        self.vehicle_count = vehicle_count
        # Simple algorithm: congestion between 0-100
        # This can be fine-tuned based on lane capacity and expected traffic
        self.congestion_level = min(100, (vehicle_count / 20) * 100)
        self.history.append(self.congestion_level)
        return self.congestion_level


class TrafficControlSystem:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.vehicle_classes = [2, 3, 5, 7]  # COCO classes for vehicles (car, motorcycle, bus, truck)
        self.pedestrian_class = 0
        self.emergency_classes = [3, 5]  # Motorcycles (police bikes), buses (ambulances)
        
        # Priority parameters
        self.vehicle_weight = 1.0
        self.pedestrian_weight = 0.5
        self.emergency_boost = 100  # Absolute priority for emergency vehicles
        self.starvation_factor = 0.2  # Priority increase per second of waiting
        self.max_starvation_boost = 5.0  # Maximum starvation boost
        
        # Lane state tracking
        self.lane_states = defaultdict(dict)
        self.current_green_lane = None
        self.last_decision_time = time.time()
        
        # Video processing attributes
        self.video_source = None
        self.lanes = []
        self.min_green_time = 10  # Minimum green light duration in seconds
        self.max_green_time = 60  # Maximum green light duration in seconds
        self.frame = None
        self.running = False
        self.vehicle_detector = VehicleDetector()
        
    def add_lane(self, name, roi_points):
        """Add a traffic lane with a region of interest"""
        lane = TrafficLane(name, roi_points)
        self.lanes.append(lane)
        return lane
        
    def set_video_source(self, video_path):
        """Set the video source file path"""
        if os.path.exists(video_path):
            self.video_source = video_path
            return True
        return False
        
    def start_video_processing(self):
        """Start the traffic control system with video processing"""
        if not self.video_source:
            print("Error: No video source specified.")
            return False
            
        self.running = True
        self.video_thread = threading.Thread(target=self._process_video)
        self.control_thread = threading.Thread(target=self._control_traffic)
        self.video_thread.daemon = True
        self.control_thread.daemon = True
        self.video_thread.start()
        self.control_thread.start()
        return True
        
    def stop_video_processing(self):
        """Stop the traffic control system"""
        self.running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=1.0)
        if hasattr(self, 'control_thread'):
            self.control_thread.join(timeout=1.0)
    
    def _process_video(self):
        """Process video stream and detect vehicles in each lane"""
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.video_source}")
            self.running = False
            return
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {frame_width}x{frame_height}")
        
        # Create window for displaying the processed video
        cv2.namedWindow('Traffic Control System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Control System', frame_width, frame_height)
            
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break
                
            self.frame = frame.copy()
            
            # Process each lane
            for lane in self.lanes:
                # Create mask for lane ROI
                mask = np.zeros_like(frame[:,:,0])
                cv2.fillPoly(mask, [lane.roi_points], 255)
                
                # Apply mask to frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Detect vehicles in the lane
                vehicles = self.vehicle_detector.detect(masked_frame)
                
                # Update lane congestion
                lane.calculate_congestion(len(vehicles))
                
                # Visualize lane and status
                self._visualize_lane(frame, lane)
            
            # Display the processed frame
            cv2.imshow('Traffic Control System', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Quit on 'q' press
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        
    def _control_traffic(self):
        """Control traffic signals based on congestion levels"""
        if not self.lanes:
            print("Error: No lanes defined.")
            return
            
        # Initialize with the first lane
        self.current_green_lane = self.lanes[0]
        self.current_green_lane.is_green = True
        start_time = time.time()
        
        while self.running:
            elapsed = time.time() - start_time
            
            # Check if minimum green time has passed
            if elapsed >= self.min_green_time:
                # Find the most congested lane
                most_congested = max(
                    [lane for lane in self.lanes if lane != self.current_green_lane],
                    key=lambda x: sum(x.history) / len(x.history) if x.history else 0
                )
                
                # If another lane is more congested or max green time reached
                if (most_congested.congestion_level > self.current_green_lane.congestion_level * 1.5 or 
                    elapsed >= self.max_green_time):
                    
                    # Switch the green signal
                    self.current_green_lane.is_green = False
                    self.current_green_lane.green_time = elapsed
                    
                    self.current_green_lane = most_congested
                    self.current_green_lane.is_green = True
                    
                    start_time = time.time()
                    print(f"Switching green light to {self.current_green_lane.name}, "
                          f"congestion: {self.current_green_lane.congestion_level:.1f}%")
            
            time.sleep(1)  # Check conditions every second
    
    def _visualize_lane(self, frame, lane):
        """Visualize lane on the frame with status information"""
        # Draw lane boundaries
        cv2.polylines(frame, [lane.roi_points], True, 
                     (0, 255, 0) if lane.is_green else (0, 0, 255), 2)
        
        # Get centroid of lane polygon
        M = cv2.moments(lane.roi_points)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Display lane info
            cv2.putText(frame, f"{lane.name}: {lane.vehicle_count} vehicles", 
                       (cx-60, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Congestion: {lane.congestion_level:.1f}%", 
                       (cx-60, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if lane.is_green:
                cv2.putText(frame, "GREEN", (cx-20, cy+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def calculate_priority(self, counts, lane_id):
        """Calculate priority score with starvation control"""
        current_time = time.time()
        
        # Emergency vehicles get absolute priority
        if counts['emergency'] > 0:
            return self.emergency_boost * counts['emergency']
        
        # Base priority calculation
        base_priority = (counts['vehicles'] * self.vehicle_weight + 
                        counts['pedestrians'] * self.pedestrian_weight)
        
        # Starvation control - increase priority for lanes waiting too long
        starvation_boost = 0
        if lane_id in self.lane_states and 'last_green_time' in self.lane_states[lane_id]:
            waiting_time = current_time - self.lane_states[lane_id]['last_green_time']
            starvation_boost = min(waiting_time * self.starvation_factor, self.max_starvation_boost)
        
        # Store components for display
        self.lane_states[lane_id]['base_priority'] = base_priority
        self.lane_states[lane_id]['starvation_boost'] = starvation_boost
        
        # Red light boost is handled separately and added later
        red_light_boost = self.lane_states[lane_id].get('additional_priority', 0)
        
        return base_priority + starvation_boost + red_light_boost
    
    def decide_green_light(self, all_counts):
        """Determine which lane gets green light based on priority"""
        current_time = time.time()
        priorities = {}
        
        # Calculate priorities for all lanes
        for lane_id, counts in all_counts.items():
            priorities[lane_id] = self.calculate_priority(counts, lane_id)
            self.lane_states[lane_id]['last_counts'] = counts
            self.lane_states[lane_id]['priority'] = priorities[lane_id]
        
        # Select lane with highest priority
        if priorities:
            green_lane = max(priorities.keys(), key=lambda x: priorities[x])
            
            # Update state information
            self.current_green_lane = green_lane
            self.lane_states[green_lane]['last_green_time'] = current_time
        else:
            green_lane = None
            
        self.last_decision_time = current_time
        
        return green_lane, priorities
    
    def process_image(self, image):
        """Process single image and return detection results"""
        results = self.model(image)
        return results[0]


class VehicleDetector:
    """Class to detect vehicles in video frames using computer vision"""
    def __init__(self, confidence_threshold=0.5):
        # Load pre-trained vehicle detection model
        # In a real implementation, you would use a proper model (YOLO, SSD, etc.)
        # For simplicity, we're using a placeholder implementation
        self.confidence_threshold = confidence_threshold
        
        # Load vehicle detection model
        try:
            # For this example, we're using a simplified approach
            # In a real system, you would use:
            # - YOLO (You Only Look Once)
            # - SSD (Single Shot MultiBox Detector) 
            # - Faster R-CNN
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=40, detectShadows=False)
        except Exception as e:
            print(f"Error loading vehicle detection model: {e}")
    
    def detect(self, frame):
        """Detect vehicles in the frame"""
        # In a real implementation, this would use the actual model
        # For simplicity, we're using basic motion detection
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Threshold to binary
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to identify vehicles
        min_area = 500  # Minimum area to be considered a vehicle
        vehicles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                vehicles.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # Placeholder confidence
                    'area': area
                })
        
        return vehicles


class AutomaticDatasetLoader:
    """Class to handle automatic loading and processing of traffic images from a dataset"""
    def __init__(self, traffic_app, dataset_path=None):
        self.traffic_app = traffic_app
        self.dataset_path = dataset_path
        self.running = False
        self.timer = None
        self.current_cycle = 0
        self.history = []
        self.simulation_thread = None
        
        # Priority boost parameters
        self.red_light_boost_interval = 10  # seconds
        self.red_light_priority_increment = 1.0  # +1 priority every interval
        self.last_boost_time = time.time()
        
    def select_dataset(self):
        """Open dialog to select dataset folder"""
        folder_path = filedialog.askdirectory(
            title="Select Dataset Folder Containing Traffic Images"
        )
        if folder_path:
            self.dataset_path = folder_path
            return True
        return False
        
    def load_random_images(self, num_lanes=4):
        """Load random images from dataset for each lane"""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            messagebox.showerror("Error", "Invalid dataset path. Please select a valid dataset.")
            return False
            
        # Get all image files from dataset
        image_files = []
        for file in os.listdir(self.dataset_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(self.dataset_path, file))
        
        if len(image_files) < num_lanes:
            messagebox.showerror("Error", f"Not enough images in dataset. Need at least {num_lanes} images.")
            return False
        
        # Randomly select images for each lane
        selected_images = random.sample(image_files, min(num_lanes, len(image_files)))
        
        self.traffic_app.lane_images = []
        self.traffic_app.lane_counts = {}
        
        # Process each selected image
        for i, img_path in enumerate(selected_images):
            img = cv2.imread(img_path)
            if img is not None:
                self.traffic_app.lane_images.append(img)
                
                # Process the image to detect objects
                results = self.traffic_app.traffic_system.process_image(img)
                
                # Count vehicles, pedestrians and emergency vehicles
                vehicle_count = 0
                pedestrian_count = 0
                emergency_count = 0
                
                for box in results.boxes:
                    cls = int(box.cls[0].item())
                    if cls in self.traffic_app.traffic_system.vehicle_classes:
                        vehicle_count += 1
                    elif cls == self.traffic_app.traffic_system.pedestrian_class:
                        pedestrian_count += 1
                    if cls in self.traffic_app.traffic_system.emergency_classes:
                        emergency_count += 1
                
                # Randomly add emergency vehicle for simulation purposes occasionally
                if random.random() < 0.2:  # 20% chance of emergency vehicle
                    emergency_count = max(1, emergency_count)
                
                # Store counts with lane_id as key
                lane_id = f"Lane {i+1}"
                self.traffic_app.lane_counts[lane_id] = {
                    'vehicles': vehicle_count,
                    'pedestrians': pedestrian_count,
                    'emergency': emergency_count
                }
                
        return len(self.traffic_app.lane_images) >= 2
    
    def apply_red_light_priority_boost(self):
        """Apply priority boost to lanes with red lights"""
        current_time = time.time()
        
        # Check if it's time for a boost
        if current_time - self.last_boost_time >= self.red_light_boost_interval:
            # Apply boost to all lanes except the current green one
            current_green = self.traffic_app.traffic_system.current_green_lane
            
            for lane_id in self.traffic_app.lane_counts.keys():
                if lane_id != current_green:
                    # Track additional priority in lane_states
                    if 'additional_priority' not in self.traffic_app.traffic_system.lane_states[lane_id]:
                        self.traffic_app.traffic_system.lane_states[lane_id]['additional_priority'] = 0
                    
                    # Increment additional priority
                    self.traffic_app.traffic_system.lane_states[lane_id]['additional_priority'] += self.red_light_priority_increment
                    
                    # Log the boost for history
                    self.history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'action': f"Priority boost +{self.red_light_priority_increment} applied to {lane_id}",
                        'reason': "Red light duration penalty"
                    })
            
            # Reset timer
            self.last_boost_time = current_time
            return True
        
        return False
    
    def start_automatic_simulation(self, interval=30):
        """Start automatic simulation with specified interval between cycles"""
        if self.running:
            return
            
        if not self.dataset_path:
            if not self.select_dataset():
                return
        
        self.running = True
        self.current_cycle = 0
        self.history = []
        self.last_boost_time = time.time()
        
        # Reset additional priority for all lanes
        for lane_id in self.traffic_app.traffic_system.lane_states:
            self.traffic_app.traffic_system.lane_states[lane_id]['additional_priority'] = 0
        
        # Show initial loading status
        for widget in self.traffic_app.result_frame.winfo_children():
            widget.destroy()
            
        status_label = ttk.Label(self.traffic_app.result_frame, 
                                text="Starting automatic simulation...", 
                                style='Heading.TLabel')
        status_label.pack(expand=True)
        
        # Create a separate thread for simulation to avoid blocking UI
        self.simulation_thread = threading.Thread(target=self._run_simulation_loop, args=(interval,))
        self.simulation_thread.daemon = True  # Thread will exit when main program exits
        self.simulation_thread.start()
    
    def _run_simulation_loop(self, interval):
        """Run continuous simulation in a separate thread"""
        while self.running:
            # Load new random images
            if self.load_random_images():
                self.current_cycle += 1
                
                # Process and display results
                self.traffic_app.root.after(0, self.traffic_app.process_and_display)
                self.traffic_app.root.after(0, self._update_status_display)
                
                # Add cycle to history
                self.history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'action': f"Cycle {self.current_cycle} started",
                    'reason': "New traffic data loaded"
                })
                
                # Wait interval between full simulation cycles
                cycle_start_time = time.time()
                while self.running and (time.time() - cycle_start_time < interval):
                    # Check for priority boost every second
                    boost_applied = self.apply_red_light_priority_boost()
                    if boost_applied:
                        # Update display when boost is applied
                        self.traffic_app.root.after(0, self._update_boost_display)
                    time.sleep(1)
            else:
                # If loading fails, stop simulation
                self.stop_automatic_simulation()
                break
    
    def _update_status_display(self):
        """Update status display with cycle information"""
        # Get the notebook widget 
        for widget in self.traffic_app.result_frame.winfo_children():
            if isinstance(widget, ttk.Notebook):
                notebook = widget
                
                # Check if we already have a status tab
                status_tab_exists = False
                for tab_id in notebook.tabs():
                    if notebook.tab(tab_id, "text") == "Simulation Status":
                        status_tab_exists = True
                        status_frame = notebook._nametowidget(tab_id)
                        break
                
                if not status_tab_exists:
                    # Create new status tab
                    status_frame = ttk.Frame(notebook)
                    notebook.add(status_frame, text="Simulation Status")
                
                # Clear previous status content
                for child in status_frame.winfo_children():
                    child.destroy()
                
                # Add simulation info
                ttk.Label(status_frame, text=f"Automatic Simulation - Cycle {self.current_cycle}", 
                         font=('Helvetica', 14, 'bold')).pack(pady=10)
                
                ttk.Label(status_frame, text=f"Dataset: {self.dataset_path}", 
                         wraplength=500).pack(pady=5)
                
                ttk.Label(status_frame, text=f"Red Light Priority Boost: +{self.red_light_priority_increment} every {self.red_light_boost_interval} seconds", 
                         font=('Helvetica', 12)).pack(pady=5)
                
                # Add history in a scrollable text widget
                history_frame = ttk.LabelFrame(status_frame, text="Simulation History")
                history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                history_text = tk.Text(history_frame, height=15, width=60)
                scrollbar = ttk.Scrollbar(history_frame, command=history_text.yview)
                history_text.configure(yscrollcommand=scrollbar.set)
                
                history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Populate history
                for entry in self.history:
                    history_text.insert(tk.END, f"[{entry['time']}] {entry['action']}\n")
                    history_text.insert(tk.END, f"   Reason: {entry['reason']}\n\n")
                
                history_text.configure(state="disabled")  # Make read-only
    
    def _update_boost_display(self):
        """Update the display when a priority boost is applied"""
        # Only update the data analysis tab to show new priorities
        for widget in self.traffic_app.result_frame.winfo_children():
            if isinstance(widget, ttk.Notebook):
                notebook = widget
                
                # Find the data analysis tab
                for tab_id in notebook.tabs():
                    if notebook.tab(tab_id, "text") == "Data Analysis":
                        data_frame = notebook._nametowidget(tab_id)
                        
                        # Recalculate priorities and update display
                        all_counts = self.traffic_app.lane_counts
                        green_lane = self.traffic_app.traffic_system.current_green_lane
                        
                        # Calculate current priorities (without changing green lane)
                        priorities = {}
                        for lane_id, counts in all_counts.items():
                            priorities[lane_id] = self.traffic_app.traffic_system.calculate_priority(counts, lane_id)
                        
                        # Clear and rebuild data analysis display
                        for child in data_frame.winfo_children():
                            child.destroy()
                        
                        self.traffic_app.show_data_analysis(data_frame, all_counts, priorities, green_lane)
                        break
                
                # Also update the status tab with new history
                self._update_status_display()
    
    def stop_automatic_simulation(self):
        """Stop the automatic simulation"""
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(1.0)  # Wait for thread to finish with timeout
        
        # Add stop event to history
        if hasattr(self, 'history'):
            self.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'action': "Automatic simulation stopped",
                'reason': "User requested stop or simulation error"
            })
            
            # Update display one last time
            if hasattr(self, '_update_status_display'):
                self.traffic_app.root.after(0, self._update_status_display)


class VideoUploader:
    """Handles video file upload from local system"""
    def __init__(self, root=None):
        self.video_path = None
        self.root = root if root else tk.Tk()
        if not root:
            self.root.withdraw()  # Hide the root window if created by this class
    
    def upload_video(self):
        """Open file dialog to select a video file"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        
        self.video_path = filedialog.askopenfilename(
            title="Select Traffic Video",
            filetypes=file_types
        )
        
        if self.video_path:
            if os.path.exists(self.video_path):
                messagebox.showinfo("Success", f"Video selected: {os.path.basename(self.video_path)}")
                return self.video_path
            else:
                messagebox.showerror("Error", "File does not exist")
                return None
        return None


class TrafficControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Traffic Control System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.colors =  {
             "primary": "#1a237e",      # Dark blue
            "secondary": "#5c6bc0",    # Medium blue
            "accent": "#f50057",       # Pink accent
            "bg_light": "#f5f5f7",     # Light background
            "bg_dark": "#e0e0e0",      # Darker background
            "text_dark": "#212121",    # Dark text
            "text_light": "#ffffff",   # Light text
            "success": "#00c853",      # Green for success/go
            "warning": "#ff6d00",      # Orange for warning
            "danger": "#d50000"        # Red for danger/stop
        }
        self.configure_styles()

        self.traffic_system = TrafficControlSystem()
        self.simulator = TrafficSimulator()
        self.lane_images = []
        self.lane_counts = {}  # Using dictionary instead of list
        
        self.results = None
        
        self.setup_ui()

        # Initialize automatic dataset loader
        self.auto_loader = AutomaticDatasetLoader(self)
        
        # Initialize video uploader
        self.video_uploader = VideoUploader(self.root)

    def configure_styles(self):
        """Configure custom styles for the application"""
        # Configure global styling
        self.root.configure(bg=self.colors["bg_light"])
        
        # Create custom style
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base
        
        # Configure fonts
        title_font = ('Segoe UI', 16, 'bold')
        heading_font = ('Segoe UI', 14, 'bold')
        text_font = ('Segoe UI', 11)
        
        # Configure frame styles
        style.configure('TFrame', background=self.colors["bg_light"])
        style.configure('TLabelframe', background=self.colors["bg_light"], bordercolor=self.colors["primary"])
        style.configure('TLabelframe.Label', 
                       font=heading_font, 
                       background=self.colors["bg_light"], 
                       foreground=self.colors["primary"])
        
        # Configure label styles
        style.configure('TLabel', background=self.colors["bg_light"], font=text_font)
        style.configure('Title.TLabel', font=title_font, foreground=self.colors["primary"])
        style.configure('Heading.TLabel', font=heading_font, foreground=self.colors["secondary"])
        
        # Configure button styles
        style.configure('TButton', 
                       font=text_font, 
                       background=self.colors["primary"],
                       foreground=self.colors["text_light"])
        
        style.map('TButton', 
                 background=[('active', self.colors["secondary"])],
                 foreground=[('active', self.colors["text_light"])])
        
        # Special buttons
        style.configure('Primary.TButton', 
                       background=self.colors["primary"], 
                       foreground=self.colors["text_light"])
        
        style.configure('Accent.TButton', 
                       background=self.colors["accent"], 
                       foreground=self.colors["text_light"])
        
        # Configure notebook styles
        style.configure('TNotebook', background=self.colors["bg_light"])
        style.configure('TNotebook.Tab', 
                       font=text_font,
                       background=self.colors["bg_dark"], 
                       foreground=self.colors["text_dark"])
        
        style.map('TNotebook.Tab', 
                 background=[('selected', self.colors["primary"])],
                 foreground=[('selected', self.colors["text_light"])])
        
        # Configure treeview styles
        style.configure('Treeview', 
                       font=text_font,
                       background=self.colors["bg_light"], 
                       fieldbackground=self.colors["bg_light"])
        
        style.configure('Treeview.Heading', 
                       font=('Segoe UI', 11, 'bold'),
                       background=self.colors["secondary"], 
                       foreground=self.colors["text_light"])

    def setup_ui(self):
        """Create the application interface with auto-simulation features"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Application title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="Automated Traffic Control System", style='Title.TLabel')
        title_label.pack(anchor=tk.CENTER)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Upload section
        ttk.Label(control_frame, text="Traffic Data Input", style='Heading.TLabel').pack(pady=(5, 10))
        
        upload_btn = ttk.Button(control_frame, text="📤 Upload Traffic Images", 
                               command=self.upload_images, style='Primary.TButton')
        upload_btn.pack(fill=tk.X, pady=10, ipady=5)
        
        # Video upload button
        video_btn = ttk.Button(control_frame, text="🎥 Upload Traffic Video", 
                              command=self.upload_video, style='Primary.TButton')
        video_btn.pack(fill=tk.X, pady=10, ipady=5)
        
        # Divider
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        # Simulation section
        ttk.Label(control_frame, text="Simulation Control", style='Heading.TLabel').pack(pady=(5, 10))
        
        sim_btn = ttk.Button(control_frame, text="🔄 Run Custom Simulation", 
                            command=self.custom_simulation, style='Accent.TButton')
        sim_btn.pack(fill=tk.X, pady=10, ipady=5)
        
        # NEW: Auto-simulation controls
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        ttk.Label(control_frame, text="Automatic Dataset Processing", style='Heading.TLabel').pack(pady=(5, 10))
        
        # Dataset selection
        dataset_btn = ttk.Button(control_frame, text="📁 Select Dataset Folder", 
                                command=self.select_dataset)
        dataset_btn.pack(fill=tk.X, pady=10, ipady=5)
        
        # Automatic simulation controls
        auto_sim_frame = ttk.Frame(control_frame)
        auto_sim_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(auto_sim_frame, text="Simulation Interval (seconds):").pack(anchor=tk.W)
        
        self.sim_interval = tk.StringVar(value="30")
        interval_entry = ttk.Entry(auto_sim_frame, textvariable=self.sim_interval, width=5)
        interval_entry.pack(anchor=tk.W, pady=5)
        
        self.auto_sim_btn = ttk.Button(control_frame, text="▶️ Start Automatic Simulation", 
                                     command=self.start_auto_simulation)
        self.auto_sim_btn.pack(fill=tk.X, pady=5, ipady=5)
        
        self.stop_sim_btn = ttk.Button(control_frame, text="⏹️ Stop Simulation", 
                                     command=self.stop_auto_simulation, state=tk.DISABLED)
        self.stop_sim_btn.pack(fill=tk.X, pady=5, ipady=5)
        
        # Settings section
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        ttk.Label(control_frame, text="System Settings", style='Heading.TLabel').pack(pady=(5, 10))
        
        settings_btn = ttk.Button(control_frame, text="⚙️ Configure Parameters", 
                                 command=self.configure_parameters)
        settings_btn.pack(fill=tk.X, pady=10, ipady=5)
        
        # Right panel - Results Display Area
        self.result_frame = ttk.Frame(main_frame)
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial instruction
        instruction = ttk.Label(self.result_frame, 
                               text="Please upload traffic images or select a dataset for simulation.", 
                               style='Heading.TLabel')
        instruction.pack(expand=True)

    def upload_images(self):
        """Upload and process images for each lane"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images for Each Lane",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_paths:
            return
            
        self.lane_images = []
        self.lane_counts = {}
        
        # Process each selected image
        for i, path in enumerate(file_paths):
            img = cv2.imread(path)
            if img is not None:
                self.lane_images.append(img)
                
                # Process the image with YOLO to detect objects
                results = self.traffic_system.process_image(img)
                
                # Count vehicles, pedestrians and emergency vehicles
                vehicle_count = 0
                pedestrian_count = 0
                emergency_count = 0
                
                for box in results.boxes:
                    cls = int(box.cls[0].item())
                    if cls in self.traffic_system.vehicle_classes:
                        vehicle_count += 1
                    elif cls == self.traffic_system.pedestrian_class:
                        pedestrian_count += 1
                    if cls in self.traffic_system.emergency_classes:
                        emergency_count += 1
                
                # Store counts with lane_id as key
                lane_id = f"Lane {i+1}"
                self.lane_counts[lane_id] = {
                    'vehicles': vehicle_count,
                    'pedestrians': pedestrian_count,
                    'emergency': emergency_count
                }
        
        # Process uploaded images and display results
        if self.lane_images:
            self.process_and_display()
    
    def upload_video(self):
        """Handle video upload and setup lanes"""
        video_path = self.video_uploader.upload_video()
        if video_path:
            if self.traffic_system.set_video_source(video_path):
                self.setup_lanes_from_video(video_path)
                self.start_video_analysis()
            else:
                messagebox.showerror("Error", "Could not load video file.")
    
    def setup_lanes_from_video(self, video_path):
        """Set up traffic lanes based on video dimensions"""
        # Clear any existing lanes
        self.traffic_system.lanes = []
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video to get dimensions")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create default lanes based on video dimensions
        # North lane (top)
        self.traffic_system.add_lane("Lane 1 (North)", [
            [int(width * 0.2), 0], 
            [int(width * 0.8), 0], 
            [int(width * 0.8), int(height * 0.3)], 
            [int(width * 0.2), int(height * 0.3)]
        ])
        
        # South lane (bottom)
        self.traffic_system.add_lane("Lane 2 (South)", [
            [int(width * 0.2), int(height * 0.7)], 
            [int(width * 0.8), int(height * 0.7)], 
            [int(width * 0.8), height], 
            [int(width * 0.2), height]
        ])
        
        # East lane (right)
        self.traffic_system.add_lane("Lane 3 (East)", [
            [int(width * 0.7), int(height * 0.2)], 
            [width, int(height * 0.2)], 
            [width, int(height * 0.8)], 
            [int(width * 0.7), int(height * 0.8)]
        ])
        
        # West lane (left)
        self.traffic_system.add_lane("Lane 4 (West)", [
            [0, int(height * 0.2)], 
            [int(width * 0.3), int(height * 0.2)], 
            [int(width * 0.3), int(height * 0.8)], 
            [0, int(height * 0.8)]
        ])
        
        messagebox.showinfo("Lanes Created", 
                          f"Created 4 default lanes based on video dimensions ({width}x{height})")
    
    def start_video_analysis(self):
        """Start the video analysis"""
        if self.traffic_system.start_video_processing():
            # Clear previous results
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            # Create notebook for tabs
            notebook = ttk.Notebook(self.result_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # Create status tab
            status_frame = ttk.Frame(notebook)
            notebook.add(status_frame, text="Video Analysis Status")
            
            ttk.Label(status_frame, 
                     text="Video analysis is running in a separate window.\n"
                          "Close the video window to stop analysis.", 
                     style='Heading.TLabel').pack(pady=50)
            
            # Create a thread to monitor the video processing
            self.monitor_thread = threading.Thread(target=self._monitor_video_processing)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        else:
            messagebox.showerror("Error", "Could not start video analysis")
    
    def _monitor_video_processing(self):
        """Monitor the video processing thread"""
        while self.traffic_system.running:
            time.sleep(1)
        
        # When processing stops, update the UI
        self.root.after(0, self._video_processing_stopped)
    
    def _video_processing_stopped(self):
        """Handle video processing stopped event"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        ttk.Label(self.result_frame, 
                 text="Video analysis completed.", 
                 style='Heading.TLabel').pack(expand=True)
        
        # Generate report from the video analysis
        self.generate_video_report()
    
    def generate_video_report(self):
        """Generate a report from the video analysis"""
        if not self.traffic_system.lanes:
            return
            
        # Create notebook for tabs
        notebook = ttk.Notebook(self.result_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        
        ttk.Label(overview_frame, 
                 text="Video Analysis Summary", 
                 style="Title.TLabel").pack(pady=20)
        
        # Create a summary of lane statistics
        summary_frame = ttk.LabelFrame(overview_frame, text="Lane Statistics")
        summary_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Create table for lane details
        cols = ("Lane", "Avg Vehicles", "Max Congestion", "Total Green Time")
        lane_tree = ttk.Treeview(summary_frame, columns=cols, show="headings", height=len(self.traffic_system.lanes))
        
        # Configure columns
        for col in cols:
            lane_tree.heading(col, text=col)
            lane_tree.column(col, anchor=tk.CENTER, width=100)
        
        # Add lane data
        for lane in self.traffic_system.lanes:
            avg_vehicles = sum(lane.history) / len(lane.history) if lane.history else 0
            max_congestion = max(lane.history) if lane.history else 0
            
            lane_tree.insert("", tk.END, values=(
                lane.name,
                f"{avg_vehicles:.1f}",
                f"{max_congestion:.1f}%",
                f"{lane.green_time:.1f}s"
            ))
        
        lane_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # Add visualization button
        ttk.Button(overview_frame, text="Visualize Traffic Flow", 
                  command=self.visualize_traffic).pack(pady=20)

    def process_and_display(self):
        """Process lane images and display results"""
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()
            
        # Create notebook for tabs
        notebook = ttk.Notebook(self.result_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Calculate priorities and determine which lane gets green light
        green_lane, priorities = self.traffic_system.decide_green_light(self.lane_counts)
        
        # Create tabs
        self.create_overview_tab(notebook, green_lane)
        self.create_image_analysis_tab(notebook)
        
        # Create data analysis tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data Analysis")
        self.show_data_analysis(data_frame, self.lane_counts, priorities, green_lane)
        
        # Decision explanation tab
        self.create_decision_explanation_tab(notebook, green_lane, priorities)
        
    def create_overview_tab(self, notebook, green_lane):
        """Create an overview tab with summary information"""
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        
        # Decision summary
        ttk.Label(overview_frame, 
                text="Traffic Management Decision", 
                style="Title.TLabel").pack(pady=20)
        
        if green_lane:
            result_text = f"{green_lane} has been given priority with a GREEN light."
            color = self.colors["success"]
        else:
            result_text = "No lanes detected or insufficient data for decision."
            color = self.colors["warning"]
        
        result_label = tk.Label(overview_frame, 
                               text=result_text,
                               font=('Segoe UI', 16, 'bold'),
                               bg=self.colors["bg_light"],
                               fg=color)
        result_label.pack(pady=10)
        
        # Traffic summary
        summary_frame = ttk.LabelFrame(overview_frame, text="Traffic Summary")
        summary_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Create traffic summary table
        columns = ("Lane", "Vehicles", "Pedestrians", "Emergency")
        summary_tree = ttk.Treeview(summary_frame, columns=columns, show="headings", height=len(self.lane_counts))
        
        # Configure columns
        for col in columns:
            summary_tree.heading(col, text=col)
            summary_tree.column(col, anchor=tk.CENTER, width=100)
        
        # Add data
        total_vehicles = 0
        total_pedestrians = 0
        total_emergency = 0
        
        for lane_id, counts in self.lane_counts.items():
            summary_tree.insert("", tk.END, values=(
                lane_id, 
                counts['vehicles'], 
                counts['pedestrians'],
                counts['emergency']
            ))
            
            # Update totals
            total_vehicles += counts['vehicles']
            total_pedestrians += counts['pedestrians']
            total_emergency += counts['emergency']
        
        # Add totals row
        summary_tree.insert("", tk.END, values=("TOTAL", total_vehicles, total_pedestrians, total_emergency))
        
        summary_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # Action buttons
        action_frame = ttk.Frame(overview_frame)
        action_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(action_frame, text="Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(action_frame, text="Visualize Traffic Flow", 
                  command=self.visualize_traffic).pack(side=tk.LEFT, padx=10)
    
    def create_image_analysis_tab(self, notebook):
        """Create a tab to display image analysis for each lane"""
        image_frame = ttk.Frame(notebook)
        notebook.add(image_frame, text="Image Analysis")
        
        # Create a canvas with scrollbar for multiple images
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add canvas scrollbar
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas
        canvas = tk.Canvas(canvas_frame, 
                          xscrollcommand=h_scrollbar.set,
                          yscrollcommand=v_scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        h_scrollbar.config(command=canvas.xview)
        v_scrollbar.config(command=canvas.yview)
        
        # Create frame inside canvas to hold images
        images_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=images_frame, anchor=tk.NW)
        
        # Display images with detections
        max_width = 0
        current_y = 10
        
        for i, img in enumerate(self.lane_images):
            lane_id = f"Lane {i+1}"
            
            # Create frame for each lane
            lane_frame = ttk.LabelFrame(images_frame, text=lane_id)
            lane_frame.grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)
            
            # Process for visualization with bounding boxes
            processed_img = self.visualize_detections(img, self.traffic_system.process_image(img))
            
            # Convert for Tkinter display
            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Scale down if needed
            display_width = min(600, img.shape[1])
            scale_factor = display_width / img.shape[1]
            display_height = int(img.shape[0] * scale_factor)
            
            img_pil = img_pil.resize((display_width, display_height))
            
            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Create label to display image
            img_label = ttk.Label(lane_frame, image=img_tk)
            img_label.image = img_tk  # Keep reference
            img_label.pack(pady=5)
            
            # Image dimensions label
            dim_text = f"Image Dimensions: {img.shape[1]}x{img.shape[0]}"
            ttk.Label(lane_frame, text=dim_text).pack()
            
            # Detection counts
            counts = self.lane_counts[lane_id]
            count_text = f"Vehicles: {counts['vehicles']} | Pedestrians: {counts['pedestrians']} | Emergency: {counts['emergency']}"
            ttk.Label(lane_frame, text=count_text).pack(pady=5)
            
            # Update maximum width for proper scrolling
            current_width = display_width + 50
            max_width = max(max_width, current_width)
            current_y += display_height + 100
        
        # Update canvas scroll region
        images_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
    
    def show_data_analysis(self, frame, all_counts, priorities, green_lane):
        """Populate data analysis tab content"""
        ttk.Label(frame, text="Traffic Data Analysis", style="Title.TLabel").pack(pady=20)
        
        # Create traffic data table with advanced metrics
        columns = ("Lane", "Vehicles", "Pedestrians", "Emergency", "Base Priority", 
                  "Starvation Boost", "Red Light Boost", "Total Priority", "Status")
        data_tree = ttk.Treeview(frame, columns=columns, show="headings", height=len(all_counts))
        
        # Configure columns
        for col in columns:
            data_tree.heading(col, text=col)
            if col in ["Lane", "Status"]:
                data_tree.column(col, anchor=tk.CENTER, width=150)
            else:
                data_tree.column(col, anchor=tk.CENTER, width=100)
        
        # Add lane data with color coding
        for lane_id, counts in all_counts.items():
            lane_state = self.traffic_system.lane_states.get(lane_id, {})
            base_priority = lane_state.get('base_priority', 0)
            starvation_boost = lane_state.get('starvation_boost', 0)
            additional_priority = lane_state.get('additional_priority', 0)  # Red light boost
            
            # Calculate total priority
            total_priority = priorities.get(lane_id, 0)
            
            # Determine status
            if lane_id == green_lane:
                status = "GREEN"
                tag = "green"
            else:
                status = "RED"
                tag = "red"
            
            # Insert with appropriate tag
            data_tree.insert("", tk.END, values=(
                lane_id, 
                counts['vehicles'], 
                counts['pedestrians'],
                counts['emergency'],
                f"{base_priority:.2f}",
                f"{starvation_boost:.2f}",
                f"{additional_priority:.2f}",
                f"{total_priority:.2f}",
                status
            ), tags=(tag,))
        
        # Configure tags for color
        data_tree.tag_configure("green", background="#a5d6a7")  # Light green
        data_tree.tag_configure("red", background="#ef9a9a")    # Light red
        
        # Add table to frame with scrollbar
        tree_scroll = ttk.Scrollbar(frame, orient="vertical", command=data_tree.yview)
        data_tree.configure(yscrollcommand=tree_scroll.set)
        data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Add explanation frame
        explanation_frame = ttk.LabelFrame(frame, text="Priority Calculation Explanation")
        explanation_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Add explanation text
        explanation_text = """
Priority Calculation Formula:
• Base Priority = (Vehicle Count × Vehicle Weight) + (Pedestrian Count × Pedestrian Weight)
• Vehicle Weight = 1.0, Pedestrian Weight = 0.5
• Emergency vehicles add absolute priority (100 points per vehicle)
• Starvation Boost = Waiting Time × 0.2 (increases over time for red lights)
• Red Light Boost = Additional priority for lanes stuck at red light
• Total Priority = Base Priority + Starvation Boost + Red Light Boost

The lane with the highest Total Priority gets the green light.
        """
        
        expl_label = ttk.Label(explanation_frame, text=explanation_text, justify=tk.LEFT)
        expl_label.pack(padx=10, pady=10)
    
    def create_decision_explanation_tab(self, notebook, green_lane, priorities):
        """Create a tab explaining the decision making process"""
        decision_frame = ttk.Frame(notebook)
        notebook.add(decision_frame, text="Decision Logic")
        
        ttk.Label(decision_frame, text="Traffic Management Decision Explanation", 
                style="Title.TLabel").pack(pady=20)
        
        # Decision summary
        if green_lane:
            decision_text = f"{green_lane} received GREEN light"
            priority_value = priorities.get(green_lane, 0)
            decision_summary = f"{green_lane} was given priority with a total priority score of {priority_value:.2f}."
            
            # Get counts for the green lane
            counts = self.lane_counts.get(green_lane, {})
            vehicle_count = counts.get('vehicles', 0)
            pedestrian_count = counts.get('pedestrians', 0)
            emergency_count = counts.get('emergency', 0)
            
            # Determine primary reason for decision
            if emergency_count > 0:
                primary_reason = f"PRIMARY REASON: Emergency vehicle present in {green_lane}."
                color = self.colors["danger"]
            elif vehicle_count > 0 and vehicle_count >= pedestrian_count * 2:
                primary_reason = f"PRIMARY REASON: High vehicle density in {green_lane}."
                color = self.colors["warning"]
            elif pedestrian_count > 0 and pedestrian_count > vehicle_count:
                primary_reason = f"PRIMARY REASON: High pedestrian presence in {green_lane}."
                color = self.colors["warning"]
            else:
                primary_reason = f"PRIMARY REASON: Balanced traffic conditions, {green_lane} had highest overall priority."
                color = self.colors["secondary"]
                
            # Starvation factor
            lane_state = self.traffic_system.lane_states.get(green_lane, {})
            starvation_boost = lane_state.get('starvation_boost', 0)
            
            if starvation_boost > 0:
                starvation_text = f"Starvation factor contributed {starvation_boost:.2f} points to priority due to waiting time."
            else:
                starvation_text = "No starvation boost was applied to this lane."
        else:
            decision_text = "No traffic control decision has been made."
            decision_summary = "Insufficient data for traffic analysis."
            primary_reason = "PRIMARY REASON: No valid lanes detected or analyzed."
            starvation_text = "No starvation analysis available."
            color = self.colors["warning"]
        
        # Display decision summary
        decision_label = tk.Label(decision_frame, 
                                text=decision_text,
                                font=('Segoe UI', 16, 'bold'),
                                bg=self.colors["bg_light"],
                                fg=color)
        decision_label.pack(pady=10)
        
        summary_frame = ttk.LabelFrame(decision_frame, text="Decision Summary")
        summary_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(summary_frame, text=decision_summary, font=('Segoe UI', 12)).pack(pady=5)
        
        # Primary reason
        reason_label = ttk.Label(summary_frame, text=primary_reason, font=('Segoe UI', 12, 'bold'))
        reason_label.pack(pady=5)
        
        # Starvation information
        ttk.Label(summary_frame, text=starvation_text).pack(pady=5)
        
        # Decision process explanation
        process_frame = ttk.LabelFrame(decision_frame, text="Decision Process")
        process_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        process_text = """
The traffic management system uses the following decision process:

1. Object Detection: YOLO v8 model identifies and counts vehicles, pedestrians, and emergency vehicles in each lane.

2. Priority Calculation: For each lane, a priority score is calculated based on:
   - Vehicle count (weight: 1.0 per vehicle)
   - Pedestrian count (weight: 0.5 per pedestrian)
   - Emergency vehicles (absolute priority: 100 points each)
   - Starvation factor: Lanes waiting too long receive a boost (0.2 points per second)
   - Red light duration penalty: Additional priority for lanes stuck at red light

3. Decision: The lane with the highest priority score receives the green light.

4. Fairness Control: To prevent starvation, lanes that don't get green light accumulate additional priority over time.

5. Emergency Override: Emergency vehicles automatically get highest priority regardless of other factors.
        """
        
        # Add explanation text in scrollable text widget
        process_text_widget = tk.Text(process_frame, wrap=tk.WORD, height=15, width=70)
        process_text_widget.insert(tk.END, process_text)
        process_text_widget.configure(state="disabled")  # Make read-only
        
        scroll = ttk.Scrollbar(process_frame, command=process_text_widget.yview)
        process_text_widget.configure(yscrollcommand=scroll.set)
        
        process_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def visualize_detections(self, img, results):
        """Visualize object detections on the image"""
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Define colors for different classes
        colors = {
            'vehicle': (0, 0, 255),    # Red for vehicles
            'person': (0, 255, 0),     # Green for pedestrians
            'emergency': (0, 255, 255)  # Yellow for emergency vehicles
        }
        
        # Draw bounding boxes for detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Determine object type for color
            if cls in self.traffic_system.vehicle_classes:
                obj_type = 'vehicle'
                if cls in self.traffic_system.emergency_classes:
                    obj_type = 'emergency'
            elif cls == self.traffic_system.pedestrian_class:
                obj_type = 'person'
            else:
                obj_type = 'other'
                continue  # Skip other objects
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), colors.get(obj_type, (255, 0, 0)), 2)
            
            # Draw label with confidence
            label = f"{obj_type}: {conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(obj_type, (255, 0, 0)), 2)
        
        return vis_img
    
    def select_dataset(self):
        """Select dataset folder for automatic simulation"""
        if self.auto_loader.select_dataset():
            messagebox.showinfo("Dataset Selected", f"Dataset folder selected: {self.auto_loader.dataset_path}")
    
    def start_auto_simulation(self):
        """Start automatic simulation"""
        try:
            interval = int(self.sim_interval.get())
            if interval < 5:
                messagebox.showwarning("Invalid Interval", "Simulation interval must be at least 5 seconds.")
                return
                
            self.auto_loader.start_automatic_simulation(interval)
            self.auto_sim_btn.config(state=tk.DISABLED)
            self.stop_sim_btn.config(state=tk.NORMAL)
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for simulation interval.")
    
    def stop_auto_simulation(self):
        """Stop automatic simulation"""
        self.auto_loader.stop_automatic_simulation()
        self.auto_sim_btn.config(state=tk.NORMAL)
        self.stop_sim_btn.config(state=tk.DISABLED)
    
    def custom_simulation(self):
        """Run custom simulation with generated traffic data"""
        # Create a dialog for simulation parameters
        sim_dialog = tk.Toplevel(self.root)
        sim_dialog.title("Custom Traffic Simulation")
        sim_dialog.geometry("400x500")
        sim_dialog.grab_set()  # Make dialog modal
        
        ttk.Label(sim_dialog, text="Custom Traffic Simulation", style="Title.TLabel").pack(pady=20)
        
        # Lane count selection
        lane_frame = ttk.Frame(sim_dialog)
        lane_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(lane_frame, text="Number of Lanes:").pack(side=tk.LEFT)
        
        lane_count = tk.StringVar(value="4")
        lane_spinbox = ttk.Spinbox(lane_frame, from_=1, to=8, textvariable=lane_count, width=5)
        lane_spinbox.pack(side=tk.LEFT, padx=10)
        
        # Traffic parameters for each lane
        params_frame = ttk.LabelFrame(sim_dialog, text="Traffic Parameters")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create a canvas for scrolling if many lanes
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        
        params_content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=params_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Parameter widgets dictionary
        param_widgets = {}
        
        def update_lanes(*args):
            """Update lane parameter widgets based on lane count"""
            # Clear previous widgets
            for widget in params_content.winfo_children():
                widget.destroy()
            
            param_widgets.clear()
            
            try:
                num_lanes = int(lane_count.get())
                
                # Create parameter widgets for each lane
                for i in range(num_lanes):
                    lane_id = f"Lane {i+1}"
                    lane_label = ttk.LabelFrame(params_content, text=lane_id)
                    lane_label.pack(fill=tk.X, padx=10, pady=5)
                    
                    # Vehicle count
                    vehicle_frame = ttk.Frame(lane_label)
                    vehicle_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    ttk.Label(vehicle_frame, text="Vehicles:").pack(side=tk.LEFT)
                    vehicle_var = tk.StringVar(value=str(random.randint(0, 10)))
                    vehicle_entry = ttk.Spinbox(vehicle_frame, from_=0, to=20, textvariable=vehicle_var, width=5)
                    vehicle_entry.pack(side=tk.LEFT, padx=5)
                    
                    # Pedestrian count
                    ped_frame = ttk.Frame(lane_label)
                    ped_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    ttk.Label(ped_frame, text="Pedestrians:").pack(side=tk.LEFT)
                    ped_var = tk.StringVar(value=str(random.randint(0, 5)))
                    ped_entry = ttk.Spinbox(ped_frame, from_=0, to=15, textvariable=ped_var, width=5)
                    ped_entry.pack(side=tk.LEFT, padx=5)
                    
                    # Emergency vehicle
                    emerg_frame = ttk.Frame(lane_label)
                    emerg_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    ttk.Label(emerg_frame, text="Emergency:").pack(side=tk.LEFT)
                    emerg_var = tk.StringVar(value="0")
                    emerg_entry = ttk.Spinbox(emerg_frame, from_=0, to=2, textvariable=emerg_var, width=5)
                    emerg_entry.pack(side=tk.LEFT, padx=5)
                    
                    # Store variables for later use
                    param_widgets[lane_id] = {
                        'vehicles': vehicle_var,
                        'pedestrians': ped_var,
                        'emergency': emerg_var
                    }
                
                # Update canvas scroll region
                params_content.update_idletasks()
                canvas.config(scrollregion=canvas.bbox("all"))
                
            except ValueError:
                print("Invalid lane count")
        
        # Track changes to lane count
        lane_count.trace("w", update_lanes)
        
        # Initialize lane parameters
        update_lanes()
        
        # Button to run simulation
        def run_sim():
            try:
                # Prepare lane images and counts
                self.lane_images = []
                self.lane_counts = {}
                
                for lane_id, params in param_widgets.items():
                    vehicles_count = int(params['vehicles'].get())
                    pedestrians_count = int(params['pedestrians'].get())
                    emergency_count = int(params['emergency'].get())
                    
                    # Generate a simulated traffic image
                    vehicle_types = []
                    for _ in range(vehicles_count):
                        if emergency_count > 0:
                            vehicle_types.append('ambulance')
                            emergency_count -= 1
                        else:
                            vehicle_types.append(random.choice(['car', 'bus', 'truck', 'motorcycle']))
                    
                    # Generate image with the simulator
                    lane_img = self.simulator.generate_lane_image(
                        vehicle_types, 
                        pedestrians=pedestrians_count,
                        emergency=(emergency_count > 0)
                    )
                    
                    self.lane_images.append(lane_img)
                    
                    # Store counts
                    self.lane_counts[lane_id] = {
                        'vehicles': vehicles_count,
                        'pedestrians': pedestrians_count,
                        'emergency': int(params['emergency'].get())
                    }
                
                # Close dialog and process results
                sim_dialog.destroy()
                self.process_and_display()
                
            except Exception as e:
                messagebox.showerror("Simulation Error", f"Error running simulation: {str(e)}")
        
        # Run simulation button
        run_btn = ttk.Button(sim_dialog, text="Run Simulation", command=run_sim)
        run_btn.pack(pady=20)
    
    def configure_parameters(self):
        """Configure system parameters"""
        param_dialog = tk.Toplevel(self.root)
        param_dialog.title("System Parameters")
        param_dialog.geometry("600x500")
        param_dialog.grab_set()  # Make dialog modal
        
        ttk.Label(param_dialog, text="Traffic Management System Parameters", 
                style="Title.TLabel").pack(pady=20)
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(param_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        content_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Priority weight parameters
        weight_frame = ttk.LabelFrame(content_frame, text="Priority Weights")
        weight_frame.pack(fill=tk.X, pady=10)
        
        # Vehicle weight
        vehicle_frame = ttk.Frame(weight_frame)
        vehicle_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(vehicle_frame, text="Vehicle Weight:").pack(side=tk.LEFT)
        vehicle_weight = tk.StringVar(value=str(self.traffic_system.vehicle_weight))
        vehicle_entry = ttk.Entry(vehicle_frame, textvariable=vehicle_weight, width=10)
        vehicle_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(vehicle_frame, text="Priority points per vehicle").pack(side=tk.LEFT)
        
        # Pedestrian weight
        ped_frame = ttk.Frame(weight_frame)
        ped_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ped_frame, text="Pedestrian Weight:").pack(side=tk.LEFT)
        ped_weight = tk.StringVar(value=str(self.traffic_system.pedestrian_weight))
        ped_entry = ttk.Entry(ped_frame, textvariable=ped_weight, width=10)
        ped_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(ped_frame, text="Priority points per pedestrian").pack(side=tk.LEFT)
        
        # Emergency boost
        emerg_frame = ttk.Frame(weight_frame)
        emerg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(emerg_frame, text="Emergency Vehicle Boost:").pack(side=tk.LEFT)
        emerg_boost = tk.StringVar(value=str(self.traffic_system.emergency_boost))
        emerg_entry = ttk.Entry(emerg_frame, textvariable=emerg_boost, width=10)
        emerg_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(emerg_frame, text="Priority points per emergency vehicle").pack(side=tk.LEFT)
        
        # Starvation control parameters
        starv_frame = ttk.LabelFrame(content_frame, text="Starvation Control")
        starv_frame.pack(fill=tk.X, pady=10)
        
        # Starvation factor
        sf_frame = ttk.Frame(starv_frame)
        sf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(sf_frame, text="Starvation Factor:").pack(side=tk.LEFT)
        starv_factor = tk.StringVar(value=str(self.traffic_system.starvation_factor))
        starv_entry = ttk.Entry(sf_frame, textvariable=starv_factor, width=10)
        starv_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(sf_frame, text="Priority points per second of waiting").pack(side=tk.LEFT)
        
        # Max starvation boost
        max_starv_frame = ttk.Frame(starv_frame)
        max_starv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(max_starv_frame, text="Max Starvation Boost:").pack(side=tk.LEFT)
        max_starv = tk.StringVar(value=str(self.traffic_system.max_starvation_boost))
        max_starv_entry = ttk.Entry(max_starv_frame, textvariable=max_starv, width=10)
        max_starv_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(max_starv_frame, text="Maximum boost from starvation").pack(side=tk.LEFT)
        
        # Red light boost parameters
        red_frame = ttk.LabelFrame(content_frame, text="Red Light Priority Boost")
        red_frame.pack(fill=tk.X, pady=10)
        
        # Boost interval
        interval_frame = ttk.Frame(red_frame)
        interval_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(interval_frame, text="Boost Interval:").pack(side=tk.LEFT)
        boost_interval = tk.StringVar(value=str(self.auto_loader.red_light_boost_interval))
        interval_entry = ttk.Entry(interval_frame, textvariable=boost_interval, width=10)
        interval_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(interval_frame, text="Seconds between boosts").pack(side=tk.LEFT)
        
        # Boost increment
        increment_frame = ttk.Frame(red_frame)
        increment_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(increment_frame, text="Boost Increment:").pack(side=tk.LEFT)
        boost_inc = tk.StringVar(value=str(self.auto_loader.red_light_priority_increment))
        inc_entry = ttk.Entry(increment_frame, textvariable=boost_inc, width=10)
        inc_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(increment_frame, text="Priority points added per boost").pack(side=tk.LEFT)
        
        # Detection parameters
        detect_frame = ttk.LabelFrame(content_frame, text="Detection Settings")
        detect_frame.pack(fill=tk.X, pady=10)
        
        # Class mapping with checkboxes
        class_frame = ttk.Frame(detect_frame)
        class_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(class_frame, text="Vehicle Classes:").grid(row=0, column=0, sticky=tk.W)
        
        # YOLO class checkboxes
        class_vars = {}
        classes = {
            "Car": 2,
            "Motorcycle": 3,
            "Bus": 5,
            "Truck": 7,
            "Person": 0
        }
        
        row = 1
        for name, cls_id in classes.items():
            var = tk.BooleanVar(value=cls_id in self.traffic_system.vehicle_classes or 
                              cls_id == self.traffic_system.pedestrian_class)
            cb = ttk.Checkbutton(class_frame, text=name, variable=var)
            cb.grid(row=row, column=0, sticky=tk.W, padx=20)
            class_vars[name] = (var, cls_id)
            row += 1
        
        # Emergency vehicle classes
        ttk.Label(class_frame, text="Emergency Vehicle Classes:").grid(row=row, column=0, sticky=tk.W, pady=(10, 0))
        row += 1
        
        emerg_vars = {}
        for name, cls_id in {"Motorcycle (Police)": 3, "Bus (Ambulance)": 5}.items():
            var = tk.BooleanVar(value=cls_id in self.traffic_system.emergency_classes)
            cb = ttk.Checkbutton(class_frame, text=name, variable=var)
            cb.grid(row=row, column=0, sticky=tk.W, padx=20)
            emerg_vars[name] = (var, cls_id)
            row += 1
        
        # Update canvas scroll region
        content_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Function to save parameters
        def save_params():
            try:
                # Update priority weights
                self.traffic_system.vehicle_weight = float(vehicle_weight.get())
                self.traffic_system.pedestrian_weight = float(ped_weight.get())
                self.traffic_system.emergency_boost = float(emerg_boost.get())
                
                # Update starvation control
                self.traffic_system.starvation_factor = float(starv_factor.get())
                self.traffic_system.max_starvation_boost = float(max_starv.get())
                
                # Update red light boost parameters
                self.auto_loader.red_light_boost_interval = float(boost_interval.get())
                self.auto_loader.red_light_priority_increment = float(boost_inc.get())
                
                # Update detection classes
                vehicle_classes = []
                for name, (var, cls_id) in class_vars.items():
                    if var.get() and name != "Person":
                        vehicle_classes.append(cls_id)
                self.traffic_system.vehicle_classes = vehicle_classes
                
                # Update pedestrian class
                if class_vars["Person"][0].get():
                    self.traffic_system.pedestrian_class = class_vars["Person"][1]
                
                # Update emergency classes
                emergency_classes = []
                for name, (var, cls_id) in emerg_vars.items():
                    if var.get():
                        emergency_classes.append(cls_id)
                self.traffic_system.emergency_classes = emergency_classes
                
                param_dialog.destroy()
                messagebox.showinfo("Parameters Updated", "System parameters have been updated successfully.")
                
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter valid numeric values: {str(e)}")
        
        # Reset to defaults
        def reset_defaults():
            vehicle_weight.set("1.0")
            ped_weight.set("0.5")
            emerg_boost.set("100")
            starv_factor.set("0.2")
            max_starv.set("5.0")
            boost_interval.set("10")
            boost_inc.set("1.0")
            
            # Reset detection classes
            for name, (var, _) in class_vars.items():
                var.set(True)
            
            for name, (var, _) in emerg_vars.items():
                var.set(True)
        
        # Buttons
        btn_frame = ttk.Frame(param_dialog)
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(btn_frame, text="Save", command=save_params).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Reset to Defaults", command=reset_defaults).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=param_dialog.destroy).pack(side=tk.RIGHT, padx=10)
    
    def generate_report(self):
        """Generate detailed report of traffic analysis"""
        if not self.lane_counts:
            messagebox.showinfo("No Data", "Please run traffic analysis first.")
            return
            
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Traffic Analysis Report")
        report_window.geometry("800x600")
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(report_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        content_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Report header
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ttk.Label(content_frame, text="Traffic Management System Report", style="Title.TLabel").pack(pady=10)
        ttk.Label(content_frame, text=f"Generated: {current_time}").pack()
        
        # Summary section
        summary_frame = ttk.LabelFrame(content_frame, text="Executive Summary")
        summary_frame.pack(fill=tk.X, pady=10)
        
        green_lane = self.traffic_system.current_green_lane
        
        if green_lane:
            decision_text = f"Decision: {green_lane} has been given traffic priority."
            lane_state = self.traffic_system.lane_states.get(green_lane, {})
            last_green_time = lane_state.get('last_green_time', time.time())
            green_duration = time.time() - last_green_time
            
            summary_text = f"""
Traffic Analysis Summary:
• Number of lanes analyzed: {len(self.lane_counts)}
• Total vehicles detected: {sum(counts['vehicles'] for counts in self.lane_counts.values())}
• Total pedestrians detected: {sum(counts['pedestrians'] for counts in self.lane_counts.values())}
• Emergency vehicles detected: {sum(counts['emergency'] for counts in self.lane_counts.values())}
• Current green light: {green_lane}
• Green light duration: {green_duration:.2f} seconds
            """
        else:
            decision_text = "No traffic control decision has been made."
            summary_text = "Insufficient data for traffic analysis."
        
        ttk.Label(summary_frame, text=decision_text, font=('Segoe UI', 12, 'bold')).pack(padx=10, pady=5)
        ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT).pack(padx=10, pady=5)
        
        # Lane details
        details_frame = ttk.LabelFrame(content_frame, text="Lane Analysis Details")
        details_frame.pack(fill=tk.X, pady=10)
        
        # Create table for lane details
        cols = ("Lane", "Vehicles", "Pedestrians", "Emergency", "Priority Score", "Status")
        lane_tree = ttk.Treeview(details_frame, columns=cols, show="headings", height=len(self.lane_counts))
        
        # Configure columns
        for col in cols:
            lane_tree.heading(col, text=col)
            lane_tree.column(col, anchor=tk.CENTER, width=100)
        
        # Add lane data
        for lane_id, counts in self.lane_counts.items():
            lane_state = self.traffic_system.lane_states.get(lane_id, {})
            priority = lane_state.get('priority', 0)
            
            status = "GREEN" if lane_id == green_lane else "RED"
            tag = "green" if lane_id == green_lane else "red"
            
            lane_tree.insert("", tk.END, values=(
                lane_id,
                counts['vehicles'],
                counts['pedestrians'],
                counts['emergency'],
                f"{priority:.2f}",
                status
            ), tags=(tag,))
        
        # Configure tags for color
        lane_tree.tag_configure("green", background="#a5d6a7")
        lane_tree.tag_configure("red", background="#ef9a9a")
        
        lane_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # System parameters section
        params_frame = ttk.LabelFrame(content_frame, text="System Parameters")
        params_frame.pack(fill=tk.X, pady=10)
        
        params_text = f"""
Traffic Management System Parameters:
• Vehicle Weight: {self.traffic_system.vehicle_weight}
• Pedestrian Weight: {self.traffic_system.pedestrian_weight}
• Emergency Vehicle Boost: {self.traffic_system.emergency_boost}
• Starvation Factor: {self.traffic_system.starvation_factor} per second
• Maximum Starvation Boost: {self.traffic_system.max_starvation_boost}
• Red Light Boost Interval: {getattr(self.auto_loader, 'red_light_boost_interval', 'N/A')} seconds
• Red Light Priority Increment: {getattr(self.auto_loader, 'red_light_priority_increment', 'N/A')} points
        """
        
        ttk.Label(params_frame, text=params_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Decision process explanation
        algorithm_frame = ttk.LabelFrame(content_frame, text="Decision Algorithm")
        algorithm_frame.pack(fill=tk.X, pady=10)
        
        algorithm_text = """
The traffic management system makes decisions based on the following algorithm:

1. Object Detection: Each lane's traffic is analyzed to detect and count vehicles, pedestrians, and emergency vehicles.

2. Priority Calculation:
   - Base Priority = (Vehicle Count × Vehicle Weight) + (Pedestrian Count × Pedestrian Weight)
   - Emergency vehicles get absolute priority (Emergency Boost points per vehicle)
   - Starvation control adds priority to lanes waiting too long (Starvation Factor per second)
   - Additional priority is granted to lanes with extended red light duration

3. The lane with the highest overall priority receives the green light.

4. Priority scores are recalculated in real-time as traffic conditions change.
        """
        
        ttk.Label(algorithm_frame, text=algorithm_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Export button
        export_frame = ttk.Frame(content_frame)
        export_frame.pack(fill=tk.X, pady=20)
        
        def export_report():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Report As"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write("TRAFFIC MANAGEMENT SYSTEM REPORT\n")
                    f.write(f"Generated: {current_time}\n\n")
                    f.write("EXECUTIVE SUMMARY\n")
                    f.write("================\n")
                    f.write(f"{decision_text}\n")
                    f.write(f"{summary_text}\n\n")
                    f.write("LANE ANALYSIS DETAILS\n")
                    f.write("====================\n")
                    
                    # Write lane details
                    f.write(f"{'Lane':<10}{'Vehicles':<10}{'Pedestrians':<15}{'Emergency':<10}{'Priority':<10}{'Status':<10}\n")
                    f.write("-" * 65 + "\n")
                    
                    for lane_id, counts in self.lane_counts.items():
                        lane_state = self.traffic_system.lane_states.get(lane_id, {})
                        priority = lane_state.get('priority', 0)
                        status = "GREEN" if lane_id == green_lane else "RED"
                        
                        f.write(f"{lane_id:<10}{counts['vehicles']:<10}{counts['pedestrians']:<15}")
                        f.write(f"{counts['emergency']:<10}{priority:.2f:<10}{status:<10}\n")
                    
                    f.write("\n")
                    f.write("SYSTEM PARAMETERS\n")
                    f.write("=================\n")
                    f.write(params_text)
                    f.write("\n")
                    f.write("DECISION ALGORITHM\n")
                    f.write("=================\n")
                    f.write(algorithm_text)
                
                messagebox.showinfo("Export Complete", f"Report has been exported to {file_path}")
        
        ttk.Button(export_frame, text="Export Report", command=export_report).pack(side=tk.RIGHT)
        
        # Update canvas scroll region
        content_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    
    def generate_video_report(self):
        """Generate a report from video analysis"""
        if not hasattr(self.traffic_system, 'lanes') or not self.traffic_system.lanes:
            messagebox.showinfo("No Data", "No video analysis data available.")
            return
            
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Video Analysis Report")
        report_window.geometry("800x600")
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(report_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        content_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Report header
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ttk.Label(content_frame, text="Video Traffic Analysis Report", style="Title.TLabel").pack(pady=10)
        ttk.Label(content_frame, text=f"Generated: {current_time}").pack()
        
        # Summary section
        summary_frame = ttk.LabelFrame(content_frame, text="Executive Summary")
        summary_frame.pack(fill=tk.X, pady=10)
        
        summary_text = f"""
Video Analysis Summary:
• Number of lanes analyzed: {len(self.traffic_system.lanes)}
• Video source: {os.path.basename(self.traffic_system.video_source) if self.traffic_system.video_source else 'N/A'}
• Analysis completed at: {current_time}
        """
        
        ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Lane statistics
        stats_frame = ttk.LabelFrame(content_frame, text="Lane Statistics")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Create table for lane statistics
        cols = ("Lane", "Avg Vehicles", "Max Congestion", "Total Green Time")
        stats_tree = ttk.Treeview(stats_frame, columns=cols, show="headings", height=len(self.traffic_system.lanes))
        
        # Configure columns
        for col in cols:
            stats_tree.heading(col, text=col)
            stats_tree.column(col, anchor=tk.CENTER, width=100)
        
        # Add lane data
        for lane in self.traffic_system.lanes:
            avg_vehicles = sum(lane.history) / len(lane.history) if lane.history else 0
            max_congestion = max(lane.history) if lane.history else 0
            
            stats_tree.insert("", tk.END, values=(
                lane.name,
                f"{avg_vehicles:.1f}",
                f"{max_congestion:.1f}%",
                f"{lane.green_time:.1f}s"
            ))
        
        stats_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # Analysis parameters
        params_frame = ttk.LabelFrame(content_frame, text="Analysis Parameters")
        params_frame.pack(fill=tk.X, pady=10)
        
        params_text = f"""
Analysis Parameters:
• Minimum green time: {self.traffic_system.min_green_time} seconds
• Maximum green time: {self.traffic_system.max_green_time} seconds
• Congestion threshold: 1.5x current lane congestion
        """
        
        ttk.Label(params_frame, text=params_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Export button
        export_frame = ttk.Frame(content_frame)
        export_frame.pack(fill=tk.X, pady=20)
        
        def export_report():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Report As"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write("VIDEO TRAFFIC ANALYSIS REPORT\n")
                    f.write(f"Generated: {current_time}\n\n")
                    f.write("EXECUTIVE SUMMARY\n")
                    f.write("================\n")
                    f.write(summary_text)
                    f.write("\n")
                    f.write("LANE STATISTICS\n")
                    f.write("==============\n")
                    
                    # Write lane statistics
                    f.write(f"{'Lane':<15}{'Avg Vehicles':<15}{'Max Congestion':<15}{'Green Time':<15}\n")
                    f.write("-" * 60 + "\n")
                    
                    for lane in self.traffic_system.lanes:
                        avg_vehicles = sum(lane.history) / len(lane.history) if lane.history else 0
                        max_congestion = max(lane.history) if lane.history else 0
                        
                        f.write(f"{lane.name:<15}{avg_vehicles:<15.1f}{max_congestion:<15.1f}%{lane.green_time:<15.1f}s\n")
                    
                    f.write("\n")
                    f.write("ANALYSIS PARAMETERS\n")
                    f.write("==================\n")
                    f.write(params_text)
                
                messagebox.showinfo("Export Complete", f"Report has been exported to {file_path}")
        
        ttk.Button(export_frame, text="Export Report", command=export_report).pack(side=tk.RIGHT)
        
        # Update canvas scroll region
        content_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    
    def visualize_traffic(self):
        """Visualize traffic flow with animated simulation"""
        if not self.lane_counts and not hasattr(self.traffic_system, 'lanes'):
            messagebox.showinfo("No Data", "Please run traffic analysis first.")
            return
        
        # Create visualization window
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Traffic Flow Visualization")
        vis_window.geometry("900x700")
        
        # Main frame
        main_frame = ttk.Frame(vis_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Traffic Flow Visualization", style="Title.TLabel").pack(pady=10)
        
        # Canvas for drawing
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="white", width=800, height=500)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Start Simulation", command=lambda: start_animation()).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="Stop Simulation", command=lambda: stop_animation()).pack(side=tk.LEFT, padx=10)
        
        speed_var = tk.StringVar(value="1.0")
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(control_frame, textvariable=speed_var, width=5).pack(side=tk.LEFT)
        
        # Visualization state
        vis_state = {
            'running': False,
            'vehicles': [],
            'pedestrians': [],
            'green_lane': self.traffic_system.current_green_lane,
            'timer': None
        }
        
        # Draw the intersection
        def draw_intersection():
            canvas.delete("all")
            
            # Background
            canvas.create_rectangle(0, 0, 800, 500, fill="#555555")
            
            # Horizontal road
            canvas.create_rectangle(0, 200, 800, 300, fill="#333333")
            
            # Vertical road
            canvas.create_rectangle(350, 0, 450, 500, fill="#333333")
            
            # Lane markings
            for x in range(0, 800, 30):
                if x < 320 or x > 480:
                    canvas.create_line(x, 250, x+15, 250, fill="white", width=2)
            
            for y in range(0, 500, 30):
                if y < 170 or y > 330:
                    canvas.create_line(400, y, 400, y+15, fill="white", width=2)
            
            # Traffic lights
            light_coords = [(320, 180), (480, 320), (320, 320), (480, 180)]
            light_directions = ["North", "South", "East", "West"]
            
            for i, (x, y) in enumerate(light_coords):
                direction = light_directions[i]
                lane_id = f"Lane {i+1}"
                
                # Light box
                canvas.create_rectangle(x-10, y-30, x+10, y, fill="#222222", outline="black")
                
                # Red light
                red_color = "#777777" if lane_id == vis_state['green_lane'] else "#ff0000"
                canvas.create_oval(x-5, y-25, x+5, y-15, fill=red_color)
                
                # Green light
                green_color = "#00ff00" if lane_id == vis_state['green_lane'] else "#777777"
                canvas.create_oval(x-5, y-10, x+5, y-0, fill=green_color)
                
                # Direction label
                canvas.create_text(x, y+15, text=direction, fill="white")
        
        # Create vehicles and pedestrians based on counts
        def create_traffic():
            vis_state['vehicles'] = []
            vis_state['pedestrians'] = []
            
            # For each lane, create vehicles and pedestrians
            for i, (lane_id, counts) in enumerate(self.lane_counts.items()):
                # Determine lane direction and starting positions
                if i == 0:  # North
                    direction = "down"
                    x_range = (370, 390)
                    y_start = -50
                elif i == 1:  # South
                    direction = "up"
                    x_range = (410, 430)
                    y_start = 550
                elif i == 2:  # East
                    direction = "left"
                    x_range = (850, 850)
                    y_range = (220, 240)
                else:  # West
                    direction = "right"
                    x_range = (-50, -50)
                    y_range = (260, 280)
                
                # Create vehicles
                for j in range(counts['vehicles']):
                    if direction in ["down", "up"]:
                        x = random.randint(x_range[0], x_range[1])
                        y = y_start - j * 60 if direction == "down" else y_start + j * 60
                        
                        # Check if emergency
                        is_emergency = j < counts['emergency']
                        
                        vis_state['vehicles'].append({
                            'id': f"v{i}_{j}",
                            'x': x,
                            'y': y,
                            'direction': direction,
                            'color': "#ff0000" if is_emergency else "#3366ff",
                            'lane': lane_id,
                            'speed': 0,
                            'emergency': is_emergency
                        })
                    else:  # left or right
                        x = x_range[0] - j * 60 if direction == "left" else x_range[0] + j * 60
                        y = random.randint(y_range[0], y_range[1])
                        
                        # Check if emergency
                        is_emergency = j < counts['emergency']
                        
                        vis_state['vehicles'].append({
                            'id': f"v{i}_{j}",
                            'x': x,
                            'y': y,
                            'direction': direction,
                            'color': "#ff0000" if is_emergency else "#3366ff",
                            'lane': lane_id,
                            'speed': 0,
                            'emergency': is_emergency
                        })
                
                # Create pedestrians (only on sidewalks)
                for j in range(counts['pedestrians']):
                    if i == 0:  # North
                        x = random.randint(460, 500)
                        y = random.randint(50, 150)
                    elif i == 1:  # South
                        x = random.randint(300, 340)
                        y = random.randint(350, 450)
                    elif i == 2:  # East
                        x = random.randint(500, 600)
                        y = random.randint(150, 190)
                    else:  # West
                        x = random.randint(200, 300)
                        y = random.randint(310, 350)
                    
                    vis_state['pedestrians'].append({
                        'id': f"p{i}_{j}",
                        'x': x,
                        'y': y,
                        'direction': random.choice(["up", "down", "left", "right"]),
                        'lane': lane_id,
                        'speed': 0
                    })
        
        # Draw all traffic objects
        def draw_traffic():
            # Draw vehicles
            for vehicle in vis_state['vehicles']:
                # Draw vehicle body
                canvas.create_rectangle(
                    vehicle['x'] - 10, vehicle['y'] - 20,
                    vehicle['x'] + 10, vehicle['y'] + 20,
                    fill=vehicle['color'], outline="black", width=1
                )
                
                # Draw wheels
                canvas.create_oval(
                    vehicle['x'] - 12, vehicle['y'] - 15,
                    vehicle['x'] - 8, vehicle['y'] - 11,
                    fill="black"
                )
                canvas.create_oval(
                    vehicle['x'] + 8, vehicle['y'] - 15,
                    vehicle['x'] + 12, vehicle['y'] - 11,
                    fill="black"
                )
                canvas.create_oval(
                    vehicle['x'] - 12, vehicle['y'] + 15,
                    vehicle['x'] - 8, vehicle['y'] + 11,
                    fill="black"
                )
                canvas.create_oval(
                    vehicle['x'] + 8, vehicle['y'] + 15,
                    vehicle['x'] + 12, vehicle['y'] + 11,
                    fill="black"
                )
                
                # Draw emergency light if applicable
                if vehicle['emergency']:
                    canvas.create_oval(
                        vehicle['x'] - 5, vehicle['y'] - 25,
                        vehicle['x'] + 5, vehicle['y'] - 15,
                        fill="blue" if int(time.time() * 2) % 2 == 0 else "red"
                    )
            
            # Draw pedestrians
            for pedestrian in vis_state['pedestrians']:
                # Draw body
                canvas.create_oval(
                    pedestrian['x'] - 5, pedestrian['y'] - 5,
                    pedestrian['x'] + 5, pedestrian['y'] + 5,
                    fill="#ffaa00", outline="black"
                )
                
                # Draw stick figure
                canvas.create_line(
                    pedestrian['x'], pedestrian['y'] + 5,
                    pedestrian['x'], pedestrian['y'] + 15,
                    fill="black", width=2
                )
                canvas.create_line(
                    pedestrian['x'], pedestrian['y'] + 15,
                    pedestrian['x'] - 5, pedestrian['y'] + 25,
                    fill="black", width=2
                )
                canvas.create_line(
                    pedestrian['x'], pedestrian['y'] + 15,
                    pedestrian['x'] + 5, pedestrian['y'] + 25,
                                        fill="black", width=2
                )
                canvas.create_line(
                    pedestrian['x'] - 5, pedestrian['y'] + 10,
                    pedestrian['x'] + 5, pedestrian['y'] + 10,
                    fill="black", width=2
                )
        
        # Update traffic positions
        def update_traffic():
            speed = float(speed_var.get())
            
            for vehicle in vis_state['vehicles']:
                # Only move if green light for this lane or emergency vehicle
                if vehicle['lane'] == vis_state['green_lane'] or vehicle['emergency']:
                    if vehicle['direction'] == "down":
                        vehicle['y'] += 2 * speed
                        vehicle['speed'] = 2 * speed
                    elif vehicle['direction'] == "up":
                        vehicle['y'] -= 2 * speed
                        vehicle['speed'] = 2 * speed
                    elif vehicle['direction'] == "left":
                        vehicle['x'] -= 2 * speed
                        vehicle['speed'] = 2 * speed
                    elif vehicle['direction'] == "right":
                        vehicle['x'] += 2 * speed
                        vehicle['speed'] = 2 * speed
                else:
                    vehicle['speed'] = 0
            
            for pedestrian in vis_state['pedestrians']:
                # Pedestrians move randomly
                if random.random() < 0.3:  # 30% chance to change direction
                    pedestrian['direction'] = random.choice(["up", "down", "left", "right"])
                
                if pedestrian['direction'] == "up":
                    pedestrian['y'] -= 1 * speed
                elif pedestrian['direction'] == "down":
                    pedestrian['y'] += 1 * speed
                elif pedestrian['direction'] == "left":
                    pedestrian['x'] -= 1 * speed
                elif pedestrian['direction'] == "right":
                    pedestrian['x'] += 1 * speed
                
                # Keep pedestrians within bounds
                if pedestrian['x'] < 0: pedestrian['x'] = 0
                if pedestrian['x'] > 800: pedestrian['x'] = 800
                if pedestrian['y'] < 0: pedestrian['y'] = 0
                if pedestrian['y'] > 500: pedestrian['y'] = 500
        
        # Animation loop
        def animate():
            if vis_state['running']:
                update_traffic()
                draw_intersection()
                draw_traffic()
                vis_state['timer'] = canvas.after(50, animate)
        
        # Start animation
        def start_animation():
            if not vis_state['running']:
                vis_state['running'] = True
                vis_state['green_lane'] = self.traffic_system.current_green_lane
                draw_intersection()
                create_traffic()
                animate()
        
        # Stop animation
        def stop_animation():
            vis_state['running'] = False
            if vis_state['timer']:
                canvas.after_cancel(vis_state['timer'])
        
        # Initialize visualization
        draw_intersection()
        
        # Clean up when window closes
        vis_window.protocol("WM_DELETE_WINDOW", lambda: [stop_animation(), vis_window.destroy()])

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficControlApp(root)
    root.mainloop()