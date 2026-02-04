from google.cloud import vision
import cv2
import numpy as np
import functions_framework
from google.cloud import vision
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\User\\LLM_Construction\\llm-construction-485506-0a01a5e3cb1c.json"

# def detect_text(path, c):
#     """Detects text in the file."""
#     from google.cloud import vision

#     client = vision.ImageAnnotatorClient()

#     with open(path, "rb") as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)

#     response = client.text_detection(image=image)
#     texts = response.text_annotations
#     # print("Texts:")

#     for text in texts:
#         print(f'\n"{text.description}"')

#         vertices = [
#             f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
#         ]

#         # print("bounds: {}".format(",".join(vertices)))

#     if response.error.message:
#         raise Exception(
#             "{}\nFor more info on error messages, check: "
#             "https://cloud.google.com/apis/design/errors".format(response.error.message)
#         )
    
#     # save output to a json file
#     import json
#     output = []
#     for text in texts:
#         item = {
#             "description": text.description,
#             "bounding_poly": [
#                 {"x": vertex.x, "y": vertex.y} for vertex in text.bounding_poly.vertices
#             ],
#         }
#         output.append(item)
    
#     output_dir_json = f"C:\\Users\\User\\LLM_Construction\\dataset_json\\Method_Precast_For_Repairing_Tunnet_Segment\\{c}"
#     output_dir_bb = f"C:\\Users\\User\\LLM_Construction\\dataset_png_bb\\Method_Precast_For_Repairing_Tunnet_Segment\\{c}"
#     os.makedirs(output_dir_json, exist_ok=True)
#     os.makedirs(output_dir_bb, exist_ok=True)
#     os.makedirs(os.path.join(output_dir_bb, c), exist_ok=True)
#     # res.save_to_img(os.path.join(output_dir_bb, f"{os.path.splitext(i)[0]}.png"))
#     # res.save_to_json(os.path.join(output_dir_json, f"{os.path.splitext(i)[0]}.json"))

#     with open(os.path.join(output_dir_json, f"{os.path.splitext(os.path.basename(path))[0]}.json"), "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4, ensure_ascii=False)
    
#     # draw bounding box on the image and save it
#     image = cv2.imread(path)
#     for text in texts:
#         vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
#         cv2.polylines(image, [np.array(vertices)], isClosed=True, color=(0, 255, 0), thickness=2)
#     cv2.imwrite(os.path.join(output_dir_bb, c, f"{os.path.splitext(os.path.basename(path))[0]}.png"), image)

# def detect_text2(path):
#     """Detects text in the file."""
#     from google.cloud import vision

#     client = vision.ImageAnnotatorClient()

#     with open(path, "rb") as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)

#     response = client.text_detection(image=image)
#     texts = response.text_annotations
#     # print("Texts:")

#     for text in texts:
#         print(f'\n"{text.description}"')

#         vertices = [
#             f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
#         ]

#         # print("bounds: {}".format(",".join(vertices)))

#     if response.error.message:
#         raise Exception(
#             "{}\nFor more info on error messages, check: "
#             "https://cloud.google.com/apis/design/errors".format(response.error.message)
#         )
    
#     # save output to a json file
#     import json
#     output = []
#     for text in texts:
#         item = {
#             "description": text.description,
#             "bounding_poly": [
#                 {"x": vertex.x, "y": vertex.y} for vertex in text.bounding_poly.vertices
#             ],
#         }
#         output.append(item)
    
#     output_dir_json = f"C:\\Users\\User\\LLM_Construction\\dataset_json\\Manufactoring_and_quality_control_process"
#     output_dir_bb = f"C:\\Users\\User\\LLM_Construction\\dataset_png_bb\\Manufactoring_and_quality_control_process"
#     os.makedirs(output_dir_json, exist_ok=True)
#     os.makedirs(output_dir_bb, exist_ok=True)
#     # res.save_to_img(os.path.join(output_dir_bb, f"{os.path.splitext(i)[0]}.png"))
#     # res.save_to_json(os.path.join(output_dir_json, f"{os.path.splitext(i)[0]}.json"))

#     with open(os.path.join(output_dir_json, f"{os.path.splitext(os.path.basename(path))[0]}.json"), "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4, ensure_ascii=False)
    
#     # draw bounding box on the image and save it
#     image = cv2.imread(path)
#     for text in texts:
#         vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
#         cv2.polylines(image, [np.array(vertices)], isClosed=True, color=(0, 255, 0), thickness=2)
#     cv2.imwrite(os.path.join(output_dir_bb, f"{os.path.splitext(os.path.basename(path))[0]}.png"), image)

# if __name__ == "__main__":
#     # img_path = r"C:\\Users\\User\\LLM_Construction\\dataset_png\\Manufactoring_and_quality_control_process\\"
#     # img = os.listdir(img_path)
#     # for i in img:
#     #     print(f"Processing image: {i}")
#     #     detect_text2(os.path.join(img_path, i))

#     imgs_datasets_path = "C:\\Users\\User\\LLM_Construction\\dataset_png\\Method_Precast_For_Repairing_Tunnet_Segment"
#     content_path = os.listdir(imgs_datasets_path)  # process only first 2 folders for testing
#     for c in content_path:
#         print(f"Processing folder: {c}")
#         img = os.listdir(os.path.join(imgs_datasets_path, c))
#         for i in img:
#             img_path = os.path.join(imgs_datasets_path, c, i)
#             print(f"Processing image: {img_path}")
#             detect_text(img_path, c)