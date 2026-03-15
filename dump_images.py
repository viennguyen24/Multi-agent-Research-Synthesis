import sqlite3
import base64
import os

def dump_images() -> None:
    output_dir = "extracted_viewer"
    os.makedirs(output_dir, exist_ok=True)
    
    with sqlite3.connect('processor.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, mime_type, base64_data FROM images")
        count = 0
        for row in cursor.fetchall():
            img_id, mime_type, b64_data = row
            ext = "png"
            if mime_type == "image/jpeg":
                ext = "jpg"
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
                
            try:
                img_bytes = base64.b64decode(b64_data)
                filepath = os.path.join(output_dir, f"{img_id}.{ext}")
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                count += 1
            except Exception as e:
                print(f"Failed to dump {img_id}: {e}")
                
        print(f"Successfully extracted {count} images to {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    dump_images()
