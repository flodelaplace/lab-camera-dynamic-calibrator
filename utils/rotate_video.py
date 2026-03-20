import cv2
import argparse
import os

def rotate_video(input_path, output_path, angle):
    """
    Rotates a video by a specified angle and saves the result.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the rotated video file.
        angle (int): Rotation angle in degrees (e.g., 90, 180, 270).
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use a reliable, software-based codec instead of the original one
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # For 90/270 degrees, width and height swap
    if angle % 180 != 0:
        new_width = frame_height
        new_height = frame_width
    else:
        new_width = frame_width
        new_height = frame_height

    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        cap.release()
        return

    print(f"Rotating video '{os.path.basename(input_path)}' by {angle} degrees...")
    print(f"Original dimensions: {frame_width}x{frame_height}, New dimensions: {new_width}x{new_height}")

    # Get the rotation matrix
    center = (frame_width / 2, frame_height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust the rotation matrix to account for the change in dimensions
    if angle % 180 != 0:
        M[0, 2] += (new_width - frame_width) / 2
        M[1, 2] += (new_height - frame_height) / 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply rotation
        rotated_frame = cv2.warpAffine(frame, M, (new_width, new_height))
        out.write(rotated_frame)

    cap.release()
    out.release()
    print(f"Rotation complete. Saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate a video by a specified angle.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save the rotated video file.")
    parser.add_argument("--angle", "-a", type=int, required=True, choices=[0, 90, 180, 270],
                        help="Rotation angle in degrees (0, 90, 180, or 270).")

    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rotate_video(args.input, args.output, args.angle)
