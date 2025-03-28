import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image  # Adding PIL as an alternative
import sys
import os

def rgb_to_ycbcr(r, g, b):
    """Convert RGB values to YCbCr color space."""
    # Calculate Y (luminance)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Calculate Cb and Cr (chrominance)
    cb = b - y
    cr = r - y
    
    return y, cb, cr

def ycbcr_to_rgb(y, cb, cr):
    """Convert YCbCr values back to RGB color space."""
    # Direct calculations from the equations
    r = cr + y
    b = cb + y
    
    # Solve for G using the Y equation: Y = 0.299R + 0.587G + 0.114B
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    
    return r, g, b


def ycbcr_to_rgb_gaussian(y, cb, cr):
    """
    Convert YCbCr values back to RGB using Gaussian elimination.

    Args:
        y: Luminance component.
        cb: Blue difference chrominance component.
        cr: Red difference chrominance component.

    Returns:
        A tuple (r, g, b) representing the RGB values.
    """
    # Define the coefficients and constants
    c1, c2, c3 = 0.299, 0.587, 0.114
    
    # Construct the augmented matrix [A | d]
    # Rows correspond to equations:
    # 1: c1*R + c2*G + c3*B = Y
    # 2: 1*R + 0*G + 0*B  = Y + Cr
    # 3: 0*R + 0*G + 1*B  = Y + Cb
    augmented_matrix = np.array([
        [c1,  c2,  c3,  y],
        [1.0, 0.0, 0.0, y + cr],
        [0.0, 0.0, 1.0, y + cb]
    ], dtype=np.float64) # Use float64 for better precision

    print("Initial Augmented Matrix:")
    print(augmented_matrix)

    # --- Gaussian Elimination Steps ---

    # Step 1: Swap R1 and R2 to get a leading 1 in the first row
    augmented_matrix[[0, 1]] = augmented_matrix[[1, 0]]
    print("\nAfter swapping R1 and R2:")
    print(augmented_matrix)

    # Step 2: Eliminate the R coefficient in R2 (new R2). R2 = R2 - c1 * R1
    factor = augmented_matrix[1, 0] # This is c1 (0.299)
    augmented_matrix[1] = augmented_matrix[1] - factor * augmented_matrix[0]
    print("\nAfter R2 = R2 - c1 * R1:")
    print(augmented_matrix)
    
    # Step 3: Eliminate the B coefficient in R2. R2 = R2 - c3 * R3
    # Note: This step uses the value c3 (0.114) from the original matrix setup
    factor = c3 # Use the original coefficient
    augmented_matrix[1] = augmented_matrix[1] - factor * augmented_matrix[2]
    print("\nAfter R2 = R2 - c3 * R3:")
    print(augmented_matrix)

    # Step 4: Normalize R2. R2 = R2 / c2 (where c2 is 0.587)
    # The coefficient for G in R2 should now be c2 (0.587)
    pivot = augmented_matrix[1, 1] 
    if np.abs(pivot) < 1e-10: # Avoid division by zero
        raise ValueError("Matrix is singular or near-singular; cannot solve for G uniquely.")
    augmented_matrix[1] = augmented_matrix[1] / pivot
    print("\nAfter normalizing R2 (R2 = R2 / c2):")
    print(augmented_matrix)

    # --- Back Substitution (already done by Gauss-Jordan) ---
    # The matrix is now in reduced row-echelon form.
    # The solution is in the last column.
    
    r = augmented_matrix[0, 3]
    g = augmented_matrix[1, 3]
    b = augmented_matrix[2, 3]

    return r, g, b

def rgb_to_ycbcr_matrix(r, g, b):
    """
    Convert RGB values to YCbCr using matrix multiplication.

    Args:
        r: Red component (0-1).
        g: Green component (0-1).
        b: Blue component (0-1).

    Returns:
        A tuple (y, cb, cr) representing the YCbCr values.
    """
    # Define the RGB to YCbCr transformation matrix
    transform_matrix = np.array([
        [ 0.299,  0.587,  0.114],
        [-0.299, -0.587,  0.886],
        [ 0.701, -0.587, -0.114]
    ], dtype=np.float64)

    # Create the RGB column vector
    rgb_vector = np.array([r, g, b], dtype=np.float64)

    # Perform matrix multiplication: YCbCr = Matrix * RGB
    ycbcr_vector = transform_matrix @ rgb_vector 
    # Or use: ycbcr_vector = np.dot(transform_matrix, rgb_vector)

    # Extract Y, Cb, Cr
    y = ycbcr_vector[0]
    cb = ycbcr_vector[1]
    cr = ycbcr_vector[2]

    return y, cb, cr

def demo_with_image(image_path=None):
    """
    Demonstrate RGB to YCbCr conversion with either a provided image or a gradient.
    
    Args:
        image_path: Path to an image file, or None to use gradient
    """
    # Initialize image to None so we can check if we need to create a default image
    image = None
    
    if image_path is not None:
        # Debug info about the path
        print(f"Attempting to load: {image_path}")
        print(f"Absolute path: {os.path.abspath(image_path)}")
        print(f"File exists: {os.path.exists(image_path)}")
        
        # Only try to load if the file exists
        if os.path.exists(image_path):
            # Try multiple methods to load the image
            try:
                # Method 1: Try with imageio
                print("Trying to load with imageio...")
                img = imageio.imread(image_path)
                image = img.astype(np.float32) / 255.0
                
                # Ensure image has 3 channels (RGB)
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] > 3:  # Has alpha channel
                    image = image[:, :, :3]
                    
                print(f"Successfully loaded image with imageio: {image_path}, Shape: {image.shape}")
            
            except Exception as e:
                print(f"Error loading with imageio: {e}")
                
                # Method 2: Try with PIL
                try:
                    print("Trying to load with PIL...")
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')  # Ensure it's RGB
                        img_array = np.array(img)
                        image = img_array.astype(np.float32) / 255.0
                    print(f"Successfully loaded image with PIL: {image_path}, Shape: {image.shape}")
                
                except Exception as e:
                    print(f"Error loading with PIL: {e}")
                    
                    # Display more diagnostic info
                    try:
                        print(f"File size: {os.path.getsize(image_path)} bytes")
                        with open(image_path, 'rb') as f:
                            header = f.read(20)  # Read first 20 bytes to check file type
                        print(f"File header (hex): {header.hex()}")
                    except Exception as e:
                        print(f"Error getting file details: {e}")
                    
                    print("Using default gradient instead.")
                    image = None  # Reset image to None to trigger default creation
    
    # If image is still None (either no path provided or error loading), create default
    if image is None:
        # Create a default gradient image
        width, height = 300, 200
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Create a gradient
        for i in range(height):
            for j in range(width):
                r = j / width
                g = i / height
                b = 0.5
                image[i, j] = [r, g, b]
        print("Using default gradient image")
    
    # Get height and width from the image
    height, width = image.shape[:2]
    
    # Convert to YCbCr
    ycbcr_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            y, cb, cr = rgb_to_ycbcr(r, g, b)
            ycbcr_image[i, j] = [y, cb, cr]
    
    # Convert back to RGB
    reconstructed_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            y, cb, cr = ycbcr_image[i, j]
            r, g, b = ycbcr_to_rgb(y, cb, cr)
            reconstructed_image[i, j] = [r, g, b]
    
    # Display original and components
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    # Original RGB
    axes[0, 0].imshow(np.clip(image, 0, 1))
    axes[0, 0].set_title("Original RGB")
    
    vmin_val = 0
    vmax_val = 1
    
    # Noemalized RGB channels
    axes[0, 1].imshow(image[:, :, 0], cmap='Reds', vmin = vmin_val, vmax = vmax_val)
    axes[0, 1].set_title("R Channel")
    axes[0, 2].imshow(image[:, :, 1], cmap='Greens', vmin = vmin_val, vmax = vmax_val)
    axes[0, 2].set_title("G Channel")
    axes[0, 3].imshow(image[:, :, 2], cmap='Blues', vmin = vmin_val, vmax = vmax_val)
    axes[0, 3].set_title("B Channel")
    
    # YCbCr image and channels
    axes[1, 0].imshow(np.clip(reconstructed_image, 0, 1))
    axes[1, 0].set_title("Reconstructed RGB")
    
    # YCbCr channels - normalize for better visualization
    # normalization is used as the linear transformation creates negative values which would fail to render properly
    y_min, y_max = ycbcr_image[:, :, 0].min(), ycbcr_image[:, :, 0].max()
    cb_min, cb_max = ycbcr_image[:, :, 1].min(), ycbcr_image[:, :, 1].max()
    cr_min, cr_max = ycbcr_image[:, :, 2].min(), ycbcr_image[:, :, 2].max()
    
    axes[1, 1].imshow(ycbcr_image[:, :, 0], cmap='gray')
    axes[1, 1].set_title("Y Channel (Luminance)")
    axes[1, 2].imshow(ycbcr_image[:, :, 1], cmap='gray', vmin=cb_min, vmax=cb_max)
    axes[1, 2].set_title("Cb Channel (Chrominance)")
    axes[1, 3].imshow(ycbcr_image[:, :, 2], cmap='gray', vmin=cr_min, vmax=cr_max)
    axes[1, 3].set_title("Cr Channel (Chrominance)")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display difference between original and reconstructed to verify
    abs_diff = np.abs(image - reconstructed_image)
    print(f"Max difference between original and reconstructed: {np.max(abs_diff):.8f}")
    print(f"Mean difference: {np.mean(abs_diff):.8f}")

# Example usage:
if __name__ == "__main__":
    
    # --- Test RGB to YCbCr ---
    r_val, g_val, b_val = 0.9, 0.40886, 0.8 # Use the RGB values from previous example
    print(f"Input RGB: (R={r_val:.5f}, G={g_val:.5f}, B={b_val:.5f})\n")

    # Calculate using original formula
    y_orig, cb_orig, cr_orig = rgb_to_ycbcr(r_val, g_val, b_val)
    print(f"Result (Original Formula): YCbCr = (Y={y_orig:.5f}, Cb={cb_orig:.5f}, Cr={cr_orig:.5f})")

    # Calculate using matrix multiplication
    y_matrix, cb_matrix, cr_matrix = rgb_to_ycbcr_matrix(r_val, g_val, b_val)
    print(f"Result (Matrix Method):    YCbCr = (Y={y_matrix:.5f}, Cb={cb_matrix:.5f}, Cr={cr_matrix:.5f})")

    # Check if results are close
    assert np.allclose([y_orig, cb_orig, cr_orig], [y_matrix, cb_matrix, cr_matrix]), "RGB->YCbCr results do not match!"
    print("\nRGB->YCbCr results from both methods match.")
    
    print("-" * 30) # Separator

    # --- Test YCbCr to RGB ---
    y_val, cb_val, cr_val = 0.6, 0.2, 0.3 
    print(f"\nInput YCbCr: (Y={y_val}, Cb={cb_val}, Cr={cr_val})\n")

    # Calculate using Gaussian elimination
    r_gauss, g_gauss, b_gauss = ycbcr_to_rgb_gaussian(y_val, cb_val, cr_val)
    print(f"Result (Gaussian Elimination): RGB = ({r_gauss:.5f}, {g_gauss:.5f}, {b_gauss:.5f})")

    # Calculate using the direct formula
    r_direct, g_direct, b_direct = ycbcr_to_rgb(y_val, cb_val, cr_val)
    print(f"Result (Direct Formula):       RGB = ({r_direct:.5f}, {g_direct:.5f}, {b_direct:.5f})")

    # Check if results are close
    assert np.allclose([r_gauss, g_gauss, b_gauss], [r_direct, g_direct, b_direct]), "YCbCr->RGB results do not match!"
    print("\nYCbCr->RGB results from both methods match.")
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__)) + "\demo_images"
    
    # Example: You can provide an image path or None for gradient
    demo_state = True
    while demo_state:
        # List files in the script's directory
        print("\nFiles in script directory:")
        files_in_script_dir = [f for f in os.listdir(script_dir) if os.path.isfile(os.path.join(script_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # Print available image files
        for i, file in enumerate(files_in_script_dir):
            print(f"{i+1}. {file}")
        
        use_image = input("\nEnter image filename or number from the list (or press Enter for default gradient): ").strip()
        
        if use_image == "":
            use_image = None
        else:
            # Check if input is a number referring to the list
            try:
                index = int(use_image) - 1
                if 0 <= index < len(files_in_script_dir):
                    use_image = files_in_script_dir[index]
            except ValueError:
                # Input was not a number, assume it's a filename
                pass
            
            # Make sure the path is relative to the script directory
            if use_image is not None:
                use_image = os.path.join(script_dir, use_image)

        demo_with_image(use_image)
        
        answer = input("To continue press Enter, to end input any string...").strip()
        if answer != "":
            demo_state = False