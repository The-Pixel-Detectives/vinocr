import os
import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import glob

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model_name = "5CD-AI/Vintern-1B-v2"

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# Function to ensure that the directory exists
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Function to load and display the image
def display_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()

# Function to process images and save responses to text files
def process_images_and_save_responses(folder_path, output_folder, model, tokenizer, max_num=6, generation_config=None):
    # Ensure the output folder exists
    ensure_dir(output_folder)

    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Skip files that contain '._' in their names (usually macOS metadata files)
            if "._" in file_name:
                # print(f"Skipping {file_name} (contains '._').")
                continue

            # Only process files that end with .jpeg (you can modify this to include other formats if needed)
            if file_name.endswith('.jpeg'):
                # Full path to the input image
                image_path = os.path.join(root, file_name)

                # Construct corresponding output path for the text file, maintaining the folder structure
                relative_path = os.path.relpath(image_path, folder_path)
                relative_dir = os.path.dirname(relative_path)
                output_dir = os.path.join(output_folder, relative_dir)
                ensure_dir(output_dir)

                # Create output file path by replacing .jpeg with .txt
                output_txt_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')

                # **Skip processing if the output already exists**
                if os.path.exists(output_txt_path):
                    # print(f"Skipping {file_name}, output already exists.")
                    continue

                # Display the image (optional)
                # display_image(image_path)

                # Load and preprocess the image
                pixel_values = load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()

                # Set up the default generation config if not provided
                if generation_config is None:
                    generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)

                # Define the question or prompt
                question = '<image>\nMiêu tả và trích xuất tất cả kí tự, chữ, số trên hình.'

                # Start time measurement for inference
                start_time = time.time()

                # Generate the response from the model
                response = model.chat(tokenizer, pixel_values, question, generation_config)

                # End time measurement for inference
                end_time = time.time()

                # Calculate inference time
                inference_time = end_time - start_time

                # Print the response to the console
                print(f"Response for {file_name}:")
                print(response)

                # Write only the response to a text file
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(response)

                # Print the completed file and inference time
                print(f"Processed file: {file_name} and saved to: {output_txt_path}")
                print(f"Inference time for {file_name}: {inference_time:.2f} seconds")

input_folder = "/keyframes"
output_folder = "/transcriptions"

# Call the function to process all images in the folder and save transcriptions
process_images_and_save_responses(input_folder, output_folder, model, tokenizer)
