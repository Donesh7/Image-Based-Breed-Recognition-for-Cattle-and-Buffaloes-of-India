from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os
import imagehash
import shutil
from sklearn.model_selection import train_test_split

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

# Map breed keys to your folder names
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

# Paths
base_path = "DQAI_3/datasets"
raw_path = os.path.join(base_path, "raw")  # temporary storage for downloads
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")

# Settings
images_per_keyword = 50
val_split = 0.2  # 20% validation

os.makedirs(raw_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Step 1: Download images into raw/
for breed_key, keywords in breed_keywords.items():
    breed_folder = os.path.join(raw_path, folder_map[breed_key])
    os.makedirs(breed_folder, exist_ok=True)
    
    print(f"Downloading {breed_key} into {breed_folder} ...")
    for keyword in keywords:
        crawler = GoogleImageCrawler(storage={"root_dir": breed_folder})
        crawler.crawl(keyword=keyword, max_num=images_per_keyword)

# Step 2: Resize + remove duplicates
for breed_key, folder_name in folder_map.items():
    breed_folder = os.path.join(raw_path, folder_name)
    clean_images = []
    hashes = set()
    
    for filename in os.listdir(breed_folder):
        file_path = os.path.join(breed_folder, filename)
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB").resize((224, 224))
                img_hash = imagehash.average_hash(img)
                
                if img_hash in hashes:
                    os.remove(file_path)  # duplicate
                else:
                    hashes.add(img_hash)
                    clean_images.append(file_path)
                    img.save(file_path)  # overwrite resized image
        except:
            os.remove(file_path)  # corrupt file
    
    # Step 3: Train/Val Split
    train_files, val_files = train_test_split(clean_images, test_size=val_split, random_state=42)
    
    train_folder = os.path.join(train_path, folder_name)
    val_folder = os.path.join(val_path, folder_name)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    for f in train_files:
        shutil.move(f, os.path.join(train_folder, os.path.basename(f)))
    for f in val_files:
        shutil.move(f, os.path.join(val_folder, os.path.basename(f)))

print("✅ Dataset ready! Images resized, cleaned, and split into train/ and val/.")
