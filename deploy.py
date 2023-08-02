import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def kmeans_segmentation(image, k):
    # Convert the PIL Image to numpy array
    image_np = np.array(image)

    # Convert the image to RGB (if it's grayscale)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Flatten the image to 2D array (height * width, 3)
    pixels = image_np.reshape(-1, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Replace pixel values with the corresponding cluster centers
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to original image dimensions
    segmented_image = segmented_image.reshape(image_np.shape)

    return segmented_image.astype(np.uint8)

def main():
    st.title("K-Means Image Segmentation")
    st.write("Upload an image and specify the number of segments (k) to perform k-means image segmentation.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        k = st.slider("Select the number of segments (k)", 2, 20, 5)

        # Convert the uploaded file to a PIL Image
        image = Image.open(uploaded_file)

        # Perform image segmentation
        segmented_image = kmeans_segmentation(image, k)

        # Display original and segmented images
        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_column_width=True)

        st.subheader("Segmented Image")
        st.image(segmented_image, caption=f"K-Means Segmented Image (k = {k})", use_column_width=True)

if __name__ == "__main__":
    main()
