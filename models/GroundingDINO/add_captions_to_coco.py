import json

def add_captions_to_coco(json_path):
    print(f"Processing {json_path}...")
    
    # Load file JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Buat mapping category_id ke nama
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Counter untuk statistik
    total_images = len(data['images'])
    
    # Untuk setiap image, tambahkan caption
    for img in data['images']:
        img_id = img['id']
        
        # Cari semua anotasi untuk image ini
        img_annotations = [ann for ann in data['annotations'] 
                          if ann['image_id'] == img_id]
        
        # Ambil kategori unik
        categories = set([ann['category_id'] for ann in img_annotations])
        
        # Buat caption dari nama kategori
        if categories:
            cat_names = [cat_id_to_name[cat_id] for cat_id in categories if cat_id in cat_id_to_name]
            caption = '. '.join(sorted(cat_names)) + '.'
        else:
            # Jika tidak ada anotasi, buat caption kosong
            caption = "no object."
        
        # Tambahkan caption ke image
        img['caption'] = caption
    
    # Simpan kembali
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Selesai! {total_images} images telah ditambahkan caption")
    print(f"  File disimpan: {json_path}\n")

# Jalankan untuk train dan valid
add_captions_to_coco('data/tugas-akhir/train/_annotations.coco.json')
add_captions_to_coco('data/tugas-akhir/valid/_annotations.coco.json')

print("=" * 50)
print("SEMUA CAPTION BERHASIL DITAMBAHKAN!")
print("Sekarang Anda bisa menjalankan training.")
print("=" * 50)