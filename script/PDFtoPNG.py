# pdf to png conversion tool
from pdf2image import convert_from_path
import os

# Method_Precast_For_Repairing_Tunnet_Segment
def pdf_to_png(input_pdf_path, output_folder, dpi=300):
    """
    Convert a PDF file to PNG images, one image per page.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_folder (str): Folder where the PNG images will be saved.
        dpi (int): Resolution of the output images in dots per inch.

    Returns:
        List of file paths to the generated PNG images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    content = [1, 9, 15, 24, 32, 38, 44, 120-1, 124-1, 134-1, 139-1, 143-1, 149-1]
    content_name = [
        "1_Guildline_For_Repair",
        "2_Repair_With_Cementitious_Mortar",
        "3_Repair_With_Epoxy_Resin",
        "4_Spalling_Edge_Repair",
        "5_Voids_And_Blowholes_Repair",
        "6_Voids_And_Blowholes_Repair_With_High_Strength_Mortar",
        "7_Materials_Data_Sheet",
        "8_Risk_Assessment_In_Repair_Working",
        "9_Materials_Safety_Data",
        "10_Preventative_Measures_And_Revise_Environment_Impact",
        "11_Tunnel_SegmentA_Repair_Note",
        "12_Material_Approval_Document_from_the_Project_Office"
    ]

    # page87 is missing.
        
    for name in content_name:
        if not os.path.exists(os.path.join(output_folder, name)):
            os.makedirs(os.path.join(output_folder, name))

    j = 0
    while j < len(content) - 1:
        # Convert PDF to images
        images = convert_from_path(input_pdf_path, 
                                dpi=dpi, 
                                poppler_path=r'C:\\Users\\User\\poppler-25.12.0\\Library\\bin', 
                                first_page=content[j] + 4,
                                last_page=content[j+1] + 4 - 1)

        output_image_paths = []
        for i, image in enumerate(images):
            if i + content[j] >= 87:
                i += 1  # Skip page 87
            output_image_path = os.path.join(output_folder, f'{content_name[j]}', f'page_{i + content[j]}.png')
            image.save(output_image_path, 'PNG')
            output_image_paths.append(output_image_path)

        j += 1  
    return output_image_paths

# Manufactoring_and_quality_control_process
def pdf_to_png2(input_pdf_path, output_folder, dpi=300):
    # your default code here
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    # Convert PDF to images
    images = convert_from_path(input_pdf_path,
                                dpi=dpi,
                                poppler_path=r'C:\\Users\\User\\poppler-25.12.0\\Library\\bin')
    output_image_paths = []
    for i, image in enumerate(images):
        output_image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        image.save(output_image_path, 'PNG')
        output_image_paths.append(output_image_path)
    return output_image_paths

if __name__ == "__main__":
    # input_pdf = ".\\dataset_pdf\\Method_Precast_For_Repairing_Tunnet_Segment_reduced.pdf"  # Replace with your PDF file path
    # output_dir = ".\\dataset_png\\Method_Precast_For_Repairing_Tunnet_Segment\\"  # Replace with your desired output folder
    # png_files = pdf_to_png(input_pdf, output_dir)

    input_pdf = ".\\dataset_pdf\\Manufactoring_and_quality_control_process_reduced.pdf"  # Replace with your PDF file path
    output_dir = ".\\dataset_png\\Manufactoring_and_quality_control_process\\"  # Replace with your desired output folder
    png_files = pdf_to_png2(input_pdf, output_dir)