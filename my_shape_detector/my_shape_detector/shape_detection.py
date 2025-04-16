#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from shape_info.msg import ShapeInfo
import cv2
import numpy as np
import time
import threading 
import os
import datetime

class ShapeDetectionNode(Node):
    def __init__(self):
        super().__init__('shape_detection_node')
        # Initialize the OpenCV bridge
        self.declare_parameter('area_threshold', 1000)
        self.area_threshold = self.get_parameter('area_threshold').value

        # Initialize the OpenCV bridge
        self.bridge = CvBridge()

        # Create a subscriber to the input image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        # Create publishers for the output image and masks
        self.output_pub = self.create_publisher(Image, '/shape_detection/output_image', 10) 
        self.publisher = self.create_publisher(Image, '/camera/image_bgra', 10)
        self.blue_mask_pub = self.create_publisher(Image, 'blue_mask', 10)
        self.red_mask_pub = self.create_publisher(Image, 'red_mask', 10)
        self.yellow_mask_pub = self.create_publisher(Image, 'yellow_mask', 10)
        self.black_mask_pub = self.create_publisher(Image, 'black_mask', 10)
        self.shape_pub = self.create_publisher(ShapeInfo, 'shape_info', 10)
        
        # timer 0.03 ~ 30fps
        self.timer_period = 0.066
        self.fps = 0.0
        
        self.last_time = self.get_clock().now()
        
        # Add these new attributes for image capture
        self.capture_enabled = False
        # Use absolute path instead of ~ to avoid expansion issues
        self.capture_folder = "/home/nhut/quad_ws/src/hehe"
        self.capture_rate = 5  # 5 images per second
        self.capture_format = ".png"
        self.has_shape = False
        self.capture_thread = None
        self.user_name = "nam"
        self.last_capture_time = time.time()
        
        # Create capture folder if it doesn't exist
        if not os.path.exists(self.capture_folder):
            try:
                os.makedirs(self.capture_folder)
                self.get_logger().info(f'Created directory: {self.capture_folder}')
            except Exception as e:
                self.get_logger().error(f'Failed to create directory {self.capture_folder}: {e}')
        
        # Create session folder for this run
        if not hasattr(self, 'session_folder'):
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_folder = os.path.join(self.capture_folder, f"session_{current_time}")
            try:
                os.makedirs(self.session_folder, exist_ok=True)
                self.get_logger().info(f'Created session directory: {self.session_folder}')
            except Exception as e:
                self.get_logger().error(f'Failed to create session directory: {e}')
                self.session_folder = self.capture_folder
        
        self.get_logger().info('Shape Detection Node is ready to process images.')
        self.get_logger().info(f'Images will be saved to: {self.session_folder}')
        
        # Log current date/time in your format and username
        current_datetime = "2025-04-15 12:24:59"
        self.get_logger().info(f'Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_datetime}')
        self.get_logger().info(f'Current User\'s Login: {self.user_name}')
        
        # Frame counter for image capture
        self.frame_count = 0

    def image_callback(self, msg):
        """Callback function for the image subscriber"""
        # Tính thời gian xử lý
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        self.fps = 1.0 / dt if dt > 0 else 0.0
        
        try:
            # Chuyển đổi ROS Image sang OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgra8")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Tạo message cho ảnh RGB và xuất bản
            rgb_msg = self.bridge.cv2_to_imgmsg(frame_rgb, "rgb8")
            rgb_msg.header = msg.header
            self.publisher.publish(rgb_msg)
            
            # Xử lý hình ảnh
            cv_image = cv2.resize(frame_rgb, (0, 0), fx=1.0, fy=1.0)
            processed_image = self.anisotropic_diffusion(cv_image)
            
            # Phát hiện hình tròn
            blur_frame, mask_blue, mask_red, mask_yellow, mask_black, blue_on_black, red_on_black, yellow_on_black, output = self.process_frame(processed_image)
            
            # Vẽ kết quả phát hiện lên ảnh
            output, circle_info = self.draw_circles_on_frame(blur_frame, blue_on_black, red_on_black, yellow_on_black)
            
            # Determine if shapes are detected
            has_shape = False
            if ((blue_on_black is not None and len(blue_on_black) > 0) or 
                (red_on_black is not None and len(red_on_black) > 0) or 
                (yellow_on_black is not None and len(yellow_on_black) > 0) or
                (circle_info is not None and circle_info.shape_id != -1)):
                has_shape = True
            
            # Update shape detection status
            self.has_shape = has_shape
            
            # Save output image if shapes are detected and enough time has passed (rate limiting)
            current_time = time.time()
            if has_shape and (current_time - self.last_capture_time) >= (1.0 / self.capture_rate):
                self.save_output_image(output)
                self.last_capture_time = current_time
            
            # Hiển thị FPS
            self.blue_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_blue, "bgr8"))
            self.red_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_red, "bgr8"))
            self.yellow_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_yellow, "bgr8"))
            self.black_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_black, "bgr8"))
            
            # Publish output image AFTER potentially saving it
            output_msg = self.bridge.cv2_to_imgmsg(output, "bgr8")
            output_msg.header = msg.header
            self.output_pub.publish(output_msg)
            
            # Xuất bản thông tin hình dạng
            if circle_info is not None:
                self.shape_pub.publish(circle_info)
                self.get_logger().info('Publishing circle info')
            else:
                # Create an empty shape info message
                shape_info = ShapeInfo()
                shape_info.shape_id = -1
                shape_info.x = 0
                shape_info.y = 0
                shape_info.r = 0
                self.shape_pub.publish(shape_info)
                self.get_logger().info('No shape detected')
                
            # Log thông tin
            self.get_logger().debug(f'Processed frame. FPS: {self.fps:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def save_output_image(self, output_image):
        """Save the output image to the capture folder"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{self.session_folder}/{self.user_name}_output_{timestamp}{self.capture_format}"
            
            success = cv2.imwrite(filename, output_image)
            if success:
                self.frame_count += 1
                if self.frame_count % 5 == 0:  # Log every 5 frames
                    self.get_logger().info(f'Saved {self.frame_count} output images')
            else:
                self.get_logger().error(f'Failed to save output image: {filename}')
        except Exception as e:
            self.get_logger().error(f'Error saving output image: {e}')


    def start_image_capture(self):
        """Start capturing images"""
        if not self.capture_enabled:
            self.capture_enabled = True
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_images)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            self.get_logger().info(f'Image capture started - saving to {self.capture_folder}')

    def stop_image_capture(self):
        """Stop capturing images"""
        if self.capture_enabled:
            self.capture_enabled = False
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
                self.capture_thread = None
            self.get_logger().info('Image capture stopped')

    def capture_images(self):
        """Thread function to capture output images at the specified rate"""
        capture_interval = 1.0 / self.capture_rate
        frame_count = 0
        session_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = os.path.join(self.capture_folder, f"session_{session_time}")
        
        # Create session subfolder
        try:
            if not os.path.exists(session_folder):
                os.makedirs(session_folder)
                self.get_logger().info(f'Created session directory: {session_folder}')
        except Exception as e:
            self.get_logger().error(f'Failed to create session directory: {e}')
            session_folder = self.capture_folder
        
        while self.capture_enabled and self.has_shape:
            start_time = time.time()
            
            # Capture the output image if available
            with self.output_lock:
                if self.latest_output_image is not None:
                    # Generate filename with timestamp and user
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    filename = f"{session_folder}/{self.user_name}_output_{timestamp}{self.capture_format}"
                    
                    # Save the image
                    try:
                        success = cv2.imwrite(filename, self.latest_output_image)
                        
                        if success:
                            frame_count += 1
                            if frame_count % 5 == 0:  # Log every 5 frames to reduce console spam
                                self.get_logger().info(f'Saved {frame_count} output images from topic')
                        else:
                            self.get_logger().error(f'Failed to save output image: {filename}')
                    except Exception as e:
                        self.get_logger().error(f'Error saving output image: {e}')
            
            # Calculate sleep time to maintain capture rate
            elapsed = time.time() - start_time
            sleep_time = max(0, capture_interval - elapsed)
            time.sleep(sleep_time)



    def detect_blue_regions(self, frame, area_threshold=None):

        if area_threshold is None:
            area_threshold = self.area_threshold
            
        # Chuyển đổi từ BGR sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v_eq))

        # Định nghĩa ngưỡng cho màu xanh
        lower_blue = np.array([50, 70, 150])
        upper_blue = np.array([20, 30, 200])

        # Tạo mặt nạ cho vùng màu xanh
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue_colored = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)

        # Tìm các contour từ mặt nạ
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_circles = []  # Danh sách lưu trữ các vòng tròn màu xanh

        for cnt in contours:
            # Chỉ xử lý những vùng có diện tích đủ lớn
            if cv2.contourArea(cnt) > area_threshold:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                blue_circles.append([x, y, radius])
                cv2.circle(mask_blue_colored, (int(x), int(y)), int(radius), (255, 0, 0), 2)
     
        if len(blue_circles) > 0:
            blue_circles = np.array(blue_circles, dtype=np.float32)
        else:
            blue_circles = None

        return blue_circles, mask_blue_colored

    def detect_red_regions(self, frame, area_threshold=None):
        if area_threshold is None:
            area_threshold = self.area_threshold
        # Chuyển đổi từ BGR sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Định nghĩa ngưỡng cho màu đỏ
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Tạo mặt nạ cho vùng màu đỏ
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_red_colored = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)

        # Tìm các contour từ mặt nạ
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_circles = []  # Danh sách lưu trữ các vòng tròn màu đỏ

        for cnt in contours:
            # Chỉ xử lý những vùng có diện tích đủ lớn
            if cv2.contourArea(cnt) > area_threshold:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                red_circles.append([x, y, radius])
                cv2.circle(mask_red_colored, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        
        if len(red_circles) > 0:
            red_circles = np.array(red_circles, dtype=np.float32)
        else:
            red_circles = None

        return red_circles, mask_red_colored

    def detect_yellow_regions(self, frame, area_threshold=None):
        if area_threshold is None:
            area_threshold = self.area_threshold

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Định nghĩa ngưỡng cho màu vàng
        lower_yellow = np.array([20, 50, 100])
        upper_yellow = np.array([45, 255, 255])

        # Tạo mặt nạ cho vùng màu vàng
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_yellow_colored = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)

        # Tìm các contour từ mặt nạ
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_circles = []  # Danh sách lưu trữ các vòng tròn màu vàng

        for cnt in contours:
            # Chỉ xử lý những vùng có diện tích đủ lớn
            if cv2.contourArea(cnt) > area_threshold:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                yellow_circles.append([x, y, radius])
                cv2.circle(mask_yellow_colored, (int(x), int(y)), int(radius), (0, 255, 255), 2) 

        if len(yellow_circles) > 0:
            yellow_circles = np.array(yellow_circles, dtype=np.float32)
        else:
            yellow_circles = None

        return yellow_circles, mask_yellow_colored

    def detect_circles_on_black_background(self, circles_blue, circles_red, circles_yellow, frame, area_threshold=None):
        if area_threshold is None:
            area_threshold = self.area_threshold
        if frame.shape[2] == 3:  # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        # Chuyển đổi sang HSV để phát hiện màu tốt hơn
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Định nghĩa ngưỡng cho màu đen (phần nền đen mà chúng ta quan tâm)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 155, 100])  # Phạm vi rộng hơn cho màu đen
        
        # Tạo mặt nạ cho vùng màu đen
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # Áp dụng các phép biến đổi hình thái học để loại bỏ nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morphed = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.dilate(morphed, kernel, iterations=1)
        
        # Tìm kiếm các đường viền của vùng đen
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tạo một mặt nạ trống để vẽ các vùng đen được phát hiện
        mask_black_area = np.zeros_like(frame)
        
        # Xác định vùng đen lớn nhất (giả sử đó là vùng nền chính)
        black_contour = max(contours, key=cv2.contourArea) if contours else None
        
        if black_contour is not None:
            # Tạo đa giác lồi từ đường viền để có hình dạng mượt mà hơn
            hull = cv2.convexHull(black_contour)
            epsilon = 0.005 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # Vẽ vùng đen được phát hiện lên mặt nạ
            cv2.fillPoly(mask_black_area, [approx], (255, 255, 255))
            cv2.drawContours(mask_black_area, [approx], -1, (0, 255, 255), thickness=3)
            
            # Tạo mặt nạ nhị phân từ đường viền được lấp đầy
            black_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(black_mask, [approx], 255)
            
            # Tạo hình ảnh màu cho mặt nạ đen để hiển thị
            mask_black_colored = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
        else:
            approx=None
        
        # Lọc vòng tròn nằm trong vùng đen đã tìm được
        def filter_circles_with_contour(circles, contour, area_threshold=1000):
            valid = []
            if circles is not None and contour is not None:
                for (x, y, r) in circles:
                    if np.pi * (r ** 2) > area_threshold:  # Kiểm tra diện tích hình tròn
                        # Kiểm tra tâm hình tròn có nằm trong đường viền không
                        point = (int(x), int(y))
                        if cv2.pointPolygonTest(contour, point, False) >= 0:  # Nằm trong hoặc trên đường viền
                            valid.append([int(x), int(y), int(r)])
            return valid
        if approx is not None:
            valid_blue = filter_circles_with_contour(circles_blue, approx, area_threshold)
            valid_red = filter_circles_with_contour(circles_red, approx, area_threshold)
            valid_yellow = filter_circles_with_contour(circles_yellow, approx, area_threshold)
        else:
            valid_blue, valid_red, valid_yellow = None, None, None
        
        return (
            np.array(valid_blue, dtype=np.float32) if valid_blue else None,
            np.array(valid_red, dtype=np.float32) if valid_red else None,
            np.array(valid_yellow, dtype=np.float32) if valid_yellow else None,
            mask_black_colored, frame
        )

    def process_frame(self, frame_o):
        """ Blur, convert to grayscale and detect circles """
        frame = cv2.blur(frame_o, (3, 3))
        
        # Phát hiện tất cả các vòng tròn
        blue_circles, mask_blue = self.detect_blue_regions(frame)
        red_circles, mask_red = self.detect_red_regions(frame)
        yellow_circles, mask_yellow = self.detect_yellow_regions(frame)

        # Log thông tin về tất cả các vòng tròn được phát hiện
        if blue_circles is not None:
            self.get_logger().info(f'Detected {len(blue_circles)} total blue circles')
        if red_circles is not None:
            self.get_logger().info(f'Detected {len(red_circles)} total red circles')
        if yellow_circles is not None:
            self.get_logger().info(f'Detected {len(yellow_circles)} total yellow circles')
        
        # Lọc vòng tròn - chỉ lấy những vòng tròn nằm trong vùng đen
        blue_on_black, red_on_black, yellow_on_black,mask_black, output  = self.detect_circles_on_black_background(
            blue_circles, red_circles, yellow_circles, frame)
        
        # Log thông tin về các vòng tròn trong vùng đen
        if blue_on_black is not None:
            self.get_logger().info(f'Detected {len(blue_on_black)} blue circles on black background')
        if red_on_black is not None:
            self.get_logger().info(f'Detected {len(red_on_black)} red circles on black background')
        if yellow_on_black is not None:
            self.get_logger().info(f'Detected {len(yellow_on_black)} yellow circles on black background')

        return frame, mask_blue, mask_red, mask_yellow, mask_black, blue_on_black, red_on_black, yellow_on_black, output
        

    def calculate_distance(self, pointcenter, point1):
        """ Calculate the distance between two points """
        y = pointcenter[0] - point1[0]
        x = point1[1] - pointcenter[1]
        return x, y

    def get_color_name(self, r, g, b):
        """ Returns the accurate color name based on RGB values by converting to the HSV """
        color_bgr = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv

        if v < 40:
            return "Black"
        elif v > 200 and s < 40:
            return "White"
        elif s < 40:
            return "Gray"

        if (0 <= h < 25) or (h >= 140 and h <= 180):
            return "Red"
        elif 25 <= h < 85:
            return "Yellow"
        elif 85 <= h < 140:
            return "Blue"

        return "Unknown"

    def get_dominant_color_name(self, roi):
        """ Return the most dominant color name in the ROI area """
        if roi.size == 0:  # Check if ROI is empty
            return "Unknown", (0, 0, 0)
            
        roi_small = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
        pixels = roi_small.reshape(-1, 3)
        avg_color = np.mean(pixels, axis=0)
        r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
        color_name = self.get_color_name(r, g, b)
        return color_name, (b, g, r) 

    def visualize(self, image, text, unit, row_size, color_name, value):
        """ Overlay the number_dis value onto the given image. """
        if color_name == 'green':
            color = (0, 255, 0)  # RGB for green
        elif color_name == 'brown':
            color = (165, 42, 42)  # RGB for brown
        elif color_name == 'pink':
            color = (255, 192, 203)  # RGB for pink
        elif color_name == 'blue':
            color = (0, 0, 255)  # RGB for blue
        elif color_name == 'red':
            color = (255, 0, 0)  # RGB for red
        else:
            color = (255, 255, 255)  # RGB for white

        left_margin = 24
        font_size = 1
        font_thickness = 1

        text_to_display = text + ': {:.1f}'.format(value) + unit
        text_location = (left_margin, row_size)
        cv2.putText(image, text_to_display, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, color, font_thickness)
        return image
    def visualize_black_mask(self, black_mask, frame):
        """
        Hiển thị vùng đen được phát hiện lên hình ảnh gốc
        """
        # Tạo một bản sao của khung hình gốc
        viz_frame = frame.copy()
        
        # Tạo overlay màu xanh lá cây cho vùng đen được phát hiện
        overlay = np.zeros_like(viz_frame)
        overlay[black_mask > 0] = [0, 255, 0]  # Màu xanh lá
        
        # Kết hợp overlay với hình ảnh gốc
        result = cv2.addWeighted(viz_frame, 0.7, overlay, 0.3, 0)
        
        return result
    def get_circle_shape_id(self, color_name):
        """Assign shape_id for circles based on color"""
        color_name = str(color_name).lower()  # Ensure it's a string and convert to lowercase
        if "yellow" in color_name or "green" in color_name:
            return 1
        elif "red" in color_name or "orange" in color_name:
            return 2
        elif "blue" in color_name or "magenta" in color_name or "cyan" in color_name:
            return 3
        else:
            return -1  # Default ID for unrecognized colors

    def draw_circles_on_frame(self, frame, blue_circles=None, red_circles=None, yellow_circles=None):
        """ Draw the largest circle and display the dominant color """
        output = frame.copy()
        center_x = output.shape[1] // 2
        center_y = output.shape[0] // 2
        cv2.circle(output, (center_x, center_y), 4, (255, 255, 200), -1)

        # Count detected circles
        blue_count = len(blue_circles) if blue_circles is not None else 0
        red_count = len(red_circles) if red_circles is not None else 0
        yellow_count = len(yellow_circles) if yellow_circles is not None else 0
        total_circles = blue_count + red_count + yellow_count

        all_circles = []
        circle_info = None  # Initialize circle_info as None

        # Draw all blue circles with blue color
        if blue_circles is not None:
            blue_circles = np.round(blue_circles).astype("int")
            all_circles.extend(blue_circles)

        # Draw all red circles with red color
        if red_circles is not None:
            red_circles = np.round(red_circles).astype("int")
            all_circles.extend(red_circles)

        # Draw all yellow circles with yellow color
        if yellow_circles is not None:
            yellow_circles = np.round(yellow_circles).astype("int")
            all_circles.extend(yellow_circles)


        # Find and process the largest circle
        largest_circle = None
        largest_radius = 0
        shape_info = None

        for (x, y, r) in all_circles:
            if r > largest_radius:
                largest_radius = r
                largest_circle = (x, y, r)
        
        # Highlight the largest circle if found
        if largest_circle is not None:
            x, y, r = largest_circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)  # Thick green circle for largest
            
            # Highlight center of the largest circle
            cv2.circle(output, (x, y), 8, (0, 0, 255), -1)  # Larger red center dot
            cv2.drawMarker(output, (x, y), (0, 0, 255), cv2.MARKER_DIAMOND, 15, 2)  # Diamond marker
            cv2.putText(output, "LARGEST", (x - 30, y - r - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output, f"Center: ({x},{y})", (x - 50, y - r - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            # Draw direction lines
            cv2.line(output, (center_x, center_y), (x, y), (0, 0, 0), 2)  # Black line from center to circle
            cv2.line(output, (center_x, center_y), (center_x, y), (42, 42, 165), 2)  # Brown vertical line
            cv2.line(output, (x, y), (center_x, y), (255, 0, 0), 2)  # Blue horizontal line
            cv2.circle(output, (center_x, y), 4, (0, 0, 0), -1)  # Black dot at intersection

            # Calculate distance
            x11, y11 = self.calculate_distance((center_x, center_y), (x, y))

            # Display distance
            self.visualize(output, 'x1', 'px', 40, 'pink', x11)
            self.visualize(output, 'y1', 'px', 50, 'pink', y11)
            self.visualize(output, 'r', 'px', 70, 'green', r)  # Show radius

            # Create mask and get ROI for color analysis
            mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            masked_frame = cv2.bitwise_and(output, output, mask=mask)
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(output.shape[1], x + r), min(output.shape[0], y + r)


            if x1 < x2 and y1 < y2:  # Ensure ROI is valid
                roi = masked_frame[y1:y2, x1:x2]
                color_name, dominant_bgr = self.get_dominant_color_name(roi)
                cv2.putText(output, f"Circle - {color_name}", (x - 40, y - r - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, dominant_bgr, 2)
                shape_id = self.get_circle_shape_id(color_name)
                shape_info = ShapeInfo()
                shape_info.shape_id = shape_id
                shape_info.x = int(x11)
                shape_info.y = int(y11)
                shape_info.r = int(r)
                
        return output,shape_info


    def anisotropic_diffusion(self, img, num_iter=3, kappa=30, gamma=0.1):
        """
        Hàm thực hiện Anisotropic Diffusion trên hình ảnh.
        """
        img = img.astype(np.float32) / 255.0
        for _ in range(num_iter):
            dx = np.roll(img, -1, axis=1) - img
            dy = np.roll(img, -1, axis=0) - img
            c = np.exp(-(dx ** 2 + dy ** 2) / kappa)
            img += gamma * (c * (dx + dy))

        return (img * 255).astype(np.uint8)


def main(args=None):
    rclpy.init(args=args)
    shape_detection_node = ShapeDetectionNode()
    rclpy.spin(shape_detection_node)
    shape_detection_node.destroy_node()
    rclpy.shutdown()   


if __name__ == '__main__':
    main()


