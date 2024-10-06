import numpy as np
import cv2
import onnxruntime
import argparse
import warnings

warnings.filterwarnings("ignore")


def preprocess_image(image):
    img = cv2.resize(image, (256, 256))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, change to 3x256x256
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, axis=0)
    return img


def main(onnx_model_path, image_path):
    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Read and preprocess image
    image = cv2.imread(image_path)
    input_image = preprocess_image(image).astype(np.uint8)

    # Perform inference
    output = session.run([output_name], {input_name: input_image})

    # Postprocess and draw keypoints
    H, W, _ = image.shape
    for i in range(17):
        y, x, _ = output[0][0][0][i]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Save the output image
    output_image_path = "output.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Output saved to {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX model on an input image")
    parser.add_argument("onnx_model", type=str, help="Path to the ONNX model file")
    parser.add_argument("image", type=str, help="Path to the input image")
    args = parser.parse_args()

    main(args.onnx_model, args.image)
