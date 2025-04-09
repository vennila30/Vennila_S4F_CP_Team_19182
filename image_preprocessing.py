
import cv2
import numpy as np

def enhance_image(image_path, output_path='enhanced_image.jpg'):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize to 256x256
    image_resized = cv2.resize(image, (256, 256))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Histogram Equalization
    equalized = cv2.equalizeHist(denoised)
    
    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)

    # Normalize pixel values (optional, for model input)
    normalized = sharpened / 255.0

    # Save the result (convert back to 8-bit image for saving)
    output_image = np.uint8(normalized * 255)
    cv2.imwrite(output_path, output_image)
    print(f"Enhanced image saved to {output_path}")

# Example usage
# enhance_image('input.jpg')
