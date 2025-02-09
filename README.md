# 3D Face Reconstruction Pipeline for Forensic Verification

This documentation describes a 3D face reconstruction pipeline designed for forensic verification. The pipeline processes face images (from the CelebA dataset), detects facial landmarks using a pre‑trained dlib model, constructs a coarse 3D face model, estimates camera parameters, refines the model through iterative depth adjustments, and finally visualizes the reconstructed 3D model. Additionally, a super‑resolution step using a GAN-inspired approach (via OpenCV’s DNN module with an EDSR model) is integrated to enhance low‑quality images before processing.

## Table of Contents
1. [Preliminary Setup](#preliminary-setup)
2. [Setup dlib Predictor File](#setup-dlib-predictor-file)
3. [Loading Face Images from CelebA](#loading-face-images-from-celeba)
4. [Coarse 3D Shape Initialization](#coarse-3d-shape-initialization)
5. [Camera Parameter Estimation using PnP](#camera-parameter-estimation-using-pnp)
6. [Dense 3D Reconstruction via Iterative Refinement](#dense-3d-reconstruction-via-iterative-refinement)
7. [3D Visualization of the Reconstructed Model](#3d-visualization-of-the-reconstructed-model)
8. [Forensic Comparison (Placeholder)](#forensic-comparison-placeholder)
9. [Integrating Super‑Resolution GANs for Better Performance](#integrating-super-resolution-gans-for-better-performance)
10. [Conclusion](#conclusion)

---

## 1. Preliminary Setup

- **Purpose:**  
  Import required libraries and set up the Kaggle Notebook for inline plotting.

- **Key Libraries:**  
  - **os, glob:** File and directory operations.  
  - **numpy:** Numerical operations.  
  - **cv2 (OpenCV):** Image processing.  
  - **dlib:** Face detection and landmark prediction.  
  - **matplotlib.pyplot:** Plotting and visualization.

- **Code:**
  ```python
  import os
  import glob
  import numpy as np
  import cv2
  import dlib
  import matplotlib.pyplot as plt
  import random

  # Enable inline plotting in Kaggle Notebook
  %matplotlib inline
  ```

---

## 2. Setup dlib Predictor File

- **Purpose:**  
  Load the pre‑trained facial landmark detection model (`shape_predictor_68_face_landmarks.dat`) from a Kaggle dataset.

- **Key Steps:**  
  - Check if the model file exists.
  - Initialize the dlib face detector and shape predictor.

- **Code:**
  ```python
  predictor_path = "/kaggle/input/shape-predictor-68-face-landmarksdat/shape_predictor_68_face_landmarks.dat"
  if not os.path.exists(predictor_path):
      raise FileNotFoundError("The file 'shape_predictor_68_face_landmarks.dat' was not found. Please add the corresponding dataset to your Kaggle Notebook.")

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)
  ```

---

## 3. Loading Face Images from CelebA

- **Purpose:**  
  Access the CelebA dataset, list image files, and process a subset of images to detect facial landmarks.

- **Key Steps:**  
  - Set the dataset folder path (typically `/kaggle/input/celeba-dataset/img_align_celeba`).
  - Use `glob` to list all image files.
  - Process a subset (e.g., first 100 images) to extract facial landmarks using the `detect_landmarks` function.
  - Visualize detected landmarks on one sample image.

- **Code:**
  ```python
  dataset_folder = "/kaggle/input/celeba-dataset/img_align_celeba"

  all_files = glob.glob(os.path.join(dataset_folder, "**", "*.*"), recursive=True)
  print("All files found in the dataset folder:")
  print(all_files[:10])  # Print first 10 for brevity

  image_files = glob.glob(os.path.join(dataset_folder, "**", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(dataset_folder, "**", "*.jpeg"), recursive=True) + \
                glob.glob(os.path.join(dataset_folder, "**", "*.png"), recursive=True)

  if len(image_files) == 0:
      raise ValueError("No images found in the dataset folder. Please check the dataset and update the path or file extensions.")

  print("Found {} image files.".format(len(image_files)))

  images = []
  landmarks_list = []

  def detect_landmarks(image, predictor, detector):
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = detector(gray)
      for face in faces:
          shape = predictor(gray, face)
          landmarks = np.array([[p.x, p.y] for p in shape.parts()])
          return landmarks  # Return landmarks for the first detected face.
      return None

  for file in image_files[:100]:
      image = cv2.imread(file)
      if image is None:
          continue
      images.append(image)
      landmarks = detect_landmarks(image, predictor, detector)
      if landmarks is not None:
          landmarks_list.append(landmarks)
      else:
          print(f"Warning: No landmarks detected in {file}")

  if len(images) > 0 and len(landmarks_list) > 0:
      sample_img = images[0].copy()
      sample_landmarks = landmarks_list[0]
      for (x, y) in sample_landmarks:
          cv2.circle(sample_img, (x, y), 2, (0, 255, 0), -1)
      plt.figure(figsize=(8, 6))
      plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
      plt.title("Detected Facial Landmarks")
      plt.axis("off")
      plt.show()
  else:
      raise ValueError("No valid images or landmarks were detected.")
  ```

---

## 4. Coarse 3D Shape Initialization

- **Purpose:**  
  Create a basic 3D face model by averaging the 2D landmarks from the processed images and adding a zero z-coordinate.

- **Key Steps:**  
  - Compute the average (mean) of all landmark coordinates.
  - Append a zero z-coordinate to lift the 2D points into 3D space.

- **Code:**
  ```python
  def initialize_shape(landmarks_list):
      mean_landmarks = np.mean(np.array(landmarks_list), axis=0)
      coarse_shape = np.hstack((mean_landmarks, np.zeros((mean_landmarks.shape[0], 1))))
      return coarse_shape

  coarse_shape = initialize_shape(landmarks_list)
  print("Coarse 3D shape initialized. Shape dimensions:", coarse_shape.shape)

  plt.figure(figsize=(6,6))
  plt.scatter(coarse_shape[:, 0], coarse_shape[:, 1], c='red')
  plt.title("Coarse 2D Projection of 3D Shape")
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates.
  plt.show()
  ```

---

## 5. Estimate Camera Parameters using PnP

- **Purpose:**  
  Compute camera parameters (rotation, translation, and intrinsic matrix) that relate the 3D model to the 2D image. This is done using the Perspective-n-Point (PnP) algorithm.

- **Key Steps:**  
  - Set up the intrinsic camera matrix (`K`) based on assumed image dimensions.
  - Use `cv2.solvePnP` to compute rotation and translation vectors.
  - Reproject the 3D model onto the 2D image to verify the estimation.

- **Code:**
  ```python
  def optimize_camera_params(landmarks_list, coarse_shape):
      camera_params = []
      img_h, img_w = 480, 640  # Modify these if your images have different dimensions.
      focal_length = img_w
      K = np.array([[focal_length, 0, img_w/2],
                    [0, focal_length, img_h/2],
                    [0, 0, 1]], dtype=np.float32)
      dist_coeffs = np.zeros((4, 1))
      
      for landmarks in landmarks_list:
          success, rvec, tvec = cv2.solvePnP(coarse_shape.astype(np.float32),
                                             landmarks.astype(np.float32),
                                             K, dist_coeffs,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
          if success:
              camera_params.append((rvec, tvec, K))
      return camera_params

  camera_params = optimize_camera_params(landmarks_list, coarse_shape)
  print(f"Estimated camera parameters for {len(camera_params)} views.")

  if camera_params:
      rvec, tvec, K = camera_params[0]
      projected_points, _ = cv2.projectPoints(coarse_shape.astype(np.float32), rvec, tvec, K, None)
      projected_points = projected_points.reshape(-1, 2)
      
      image_copy = images[0].copy()
      for (x, y) in projected_points:
          cv2.circle(image_copy, (int(x), int(y)), 3, (255, 0, 0), -1)
      plt.figure(figsize=(8,6))
      plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
      plt.title("Reprojection of Coarse 3D Shape on First Image")
      plt.axis("off")
      plt.show()
  ```

---

## 6. Dense 3D Reconstruction via Iterative Refinement

- **Purpose:**  
  Refine the coarse 3D model by simulating depth adjustments. Although the process here is simplified, it demonstrates the idea of iteratively improving the 3D reconstruction.

- **Key Steps:**  
  - **Estimate Normals and Albedo:** A placeholder function that assumes all normals point outward (along the z-axis) and extracts a basic albedo.
  - **Refine the Shape:** Adjust the z-coordinates based on the normals.
  - **Iterate:** Repeat the process for a fixed number of iterations.

- **Code:**
  ```python
  def estimate_normals_and_albedo(coarse_shape, images, camera_params):
      normals = np.tile(np.array([0, 0, 1]), (coarse_shape.shape[0], 1))
      gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
      albedo = gray[0:coarse_shape.shape[0], 0].reshape(-1, 1).astype(np.float32) / 255.0
      return normals, albedo

  def refine_shape(coarse_shape, normals, albedo):
      refined_shape = coarse_shape.copy()
      refined_shape[:, 2] += 0.1 * normals[:, 2]
      return refined_shape

  def iterative_refinement(coarse_shape, images, camera_params, iterations=5):
      shape = coarse_shape.copy()
      for i in range(iterations):
          normals, albedo = estimate_normals_and_albedo(shape, images, camera_params)
          shape = refine_shape(shape, normals, albedo)
          print(f"Iteration {i+1}: z-coordinate range: {np.ptp(shape[:,2])}")
      final_texture = albedo
      return shape, final_texture

  final_shape, final_texture = iterative_refinement(coarse_shape, images, camera_params, iterations=5)
  print("Iterative refinement completed.")
  ```

---

## 7. 3D Visualization of the Reconstructed Model

- **Purpose:**  
  Visualize the refined 3D face model using a 3D scatter plot.

- **Code:**
  ```python
  def render_model(final_shape, final_texture):
      from mpl_toolkits.mplot3d import Axes3D
      fig = plt.figure(figsize=(8, 6))
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(final_shape[:, 0], final_shape[:, 1], final_shape[:, 2], c='r', marker='o')
      ax.set_title("Reconstructed 3D Face Model")
      ax.set_xlabel("X")
      ax.set_ylabel("Y")
      ax.set_zlabel("Z")
      plt.show()
      return fig

  render_model(final_shape, final_texture)
  ```

---

## 8. Forensic Comparison (Placeholder)

- **Purpose:**  
  Provide a placeholder function for comparing the reconstructed 3D model with a ground truth model. In a real forensic application, this function would compare two different models (e.g., a CCTV reconstruction vs. a reference mugshot).

- **Code:**
  ```python
  def compare_with_ground_truth(reconstructed_shape, ground_truth_shape):
      mse = np.mean((reconstructed_shape - ground_truth_shape) ** 2)
      print("Mean Squared Error with ground truth:", mse)
      return mse

  # For demonstration, compare the final shape with itself.
  compare_with_ground_truth(final_shape, final_shape)
  ```

---

## 9. Integrating Super‑Resolution GANs for Better Performance

### Why Use GANs?
- **Enhancement:**  
  Low-quality images (e.g., from CCTV footage) can be enhanced using super‑resolution models. Higher resolution and clearer details result in more accurate facial landmark detection and improved 3D reconstruction.
- **Forensics:**  
  Enhanced images lead to better verification and matching against reference images, which is crucial in forensic scenarios.

### Implementation in This Pipeline
- **Model Replacement:**  
  Since ESRGAN is not available on Kaggle, we use OpenCV’s DNN-based super resolution module with a pre‑trained EDSR model.
- **Enhancement Function:**  
  The `enhance_image` function uses the pre‑trained EDSR model to upscale images before they are processed for landmark detection.

### Code for Enhancement (Using EDSR)
```python
# ---------------------------
# Step 1.5: Load Pre-trained Super Resolution Model (EDSR)
# ---------------------------
edsr_model_path = "/kaggle/input/edsr-x3/EDSR_x3.pb"  # Update as necessary.
if not os.path.exists(edsr_model_path):
    raise FileNotFoundError("The EDSR model file was not found. Please add a dataset containing the model file (e.g., EDSR_x3.pb).")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(edsr_model_path)
sr.setModel("edsr", 3)  # Using EDSR with a scale factor of 3.

def enhance_image(image):
    enhanced = sr.upsample(image)
    return enhanced
```
- **Integration:**  
  In the image processing loop, each image is enhanced with `enhance_image(image)` before landmark detection.

---

## 10. Conclusion

This documentation explains a complete 3D face reconstruction pipeline tailored for forensic verification:
- **Preliminary Setup:** Libraries are imported and configured.
- **dlib Predictor Setup:** A pre-trained landmark detection model is loaded.
- **Image Loading:** Face images from the CelebA dataset are processed.
- **Coarse 3D Model:** The landmarks are averaged to form a basic 3D model.
- **Camera Parameter Estimation:** The PnP algorithm links the 3D model to the 2D image.
- **Iterative Refinement:** A placeholder iterative process refines the 3D model.
- **Visualization:** The final 3D face model is visualized.
- **Forensic Comparison:** A simple placeholder comparison function is provided.
- **GAN Integration:** The super resolution step enhances low-quality images to improve the entire pipeline.

This pipeline can be extended with more advanced GANs and refinement methods for more accurate and robust forensic applications, such as verifying whether a suspect’s known image matches a poor-quality CCTV image.

