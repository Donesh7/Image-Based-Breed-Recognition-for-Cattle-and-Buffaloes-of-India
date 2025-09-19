from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os
import imagehash

# Dictionary of breeds and their keywords
breed_keywords = {
    "Gir_cow": ["Gir cow", "Gir cattle", "Gir cow face", "Gir cow farm", "Gir cow calf", "Breed Gir cow"],
    "Sahiwal_cow": ["Sahiwal cow", "Sahiwal cattle", "Sahiwal milch cow", "Sahiwal bull", "Lambi bar cow"],
    "Tharparkar_cattle": ["Tharparkar cow", "Tharparkar cattle", "Thari cattle", "White Sindhi cattle", "Tharparkar bull"],
    "Red_Sindhi_cow": ["Red Sindhi cow", "Red Sindhi cattle", "Red Sindhi bull", "Malir cattle"],
    "Kankrej_cattle": ["Kankrej cow", "Kankrej cattle", "Kankrej bull", "Waghya cattle", "Kankrej draught"],
    "Hariana_cow": ["Hariana cow", "Hariana cattle", "Hariana bull", "Hisar cattle", "Hariana draught animal"],
    "Murrah_buffalo": ["Murrah buffalo", "Murrah black buffalo", "Murrah buffalo bull", "Murrah buffalo calf", "Delhi buffalo"],
    "Jaffarabadi_buffalo": ["Jaffarabadi buffalo", "Jaffarabadi buffalo giant", "Jaffarabadi buffalo bull", "Gir buffalo"],
    "Surti_buffalo": ["Surti buffalo", "Surti buffalo grey", "Surti buffalo with white rings", "Deccani buffalo", "Talabda buffalo"],
    "Mehsana_buffalo": ["Mehsana buffalo", "Mehsana buffalo cross", "Mehsana dairy buffalo"]
}

# Mapping dictionary keys to your existing folder names
folder_map = {
    "Gir_cow": "Gir",
    "Sahiwal_cow": "Sahiwal",
    "Tharparkar_cattle": "Tharparkar",
    "Red_Sindhi_cow": "Red_Sindhi",
    "Kankrej_cattle": "Kankrej",
    "Hariana_cow": "Hariana",
    "Murrah_buffalo": "Murrah",
    "Jaffarabadi_buffalo": "Jaffarabadi",
    "Surti_buffalo": "Surti",
    "Mehsana_buffalo": "Mehsana"
}

# Base path to your existing folders
base_path = "DQAI_3/datasets/train"
images_per_keyword = 200

for breed_key, keywords in breed_keywords.items():
    save_path = f"{base_path}/{folder_map[breed_key]}"
    print(f"Processing {breed_key} into folder {folder_map[breed_key]} ...")
    
    hashes = set()  # To track duplicate images
    
    for keyword in keywords:
        crawler = GoogleImageCrawler(storage={"root_dir": save_path})
        crawler.crawl(keyword=keyword, max_num=images_per_keyword)
    
    # Resize and remove duplicates
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        try:
            with Image.open(file_path) as img:
                # Resize to 224x224
                img = img.convert("RGB")  # ensure 3 channels
                img = img.resize((224, 224))
                
                # Check for duplicates
                img_hash = imagehash.average_hash(img)
                if img_hash in hashes:
                    os.remove(file_path)  # Remove duplicate
                else:
                    hashes.add(img_hash)
                    img.save(file_path)
        except:
            os.remove(file_path)  # Remove corrupt or unreadable images

print("Dataset is ready! All images resized and duplicates removed.")
