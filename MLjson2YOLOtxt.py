import os
import json
from PIL import Image

label_to_id = {"CAR": 0, "LPR": 1}

# input_dir = "SourceML"
input_dir = r"R:\FTP\大安\1正常"
output_dir = input_dir

# 初始化轉換計數器
total_conversions = 0

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for item in data_list:
            image_name = item["image"]
            image_path = os.path.join(input_dir, image_name)

            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                print(f"⚠️ 找不到圖片：{image_path}")
                continue

            txt_filename = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            yolo_lines = []

            for ann in item.get("annotations", []):
                label = ann["label"]
                coords = ann["coordinates"]

                if label not in label_to_id:
                    continue
                class_id = label_to_id[label]

                # 注意：此 JSON 資料 x/y 為中心點，非左上角
                # x_center = coords["x"] + int(coords["width"] / 2)
                # y_center = coords["y"] + coords["height"] / 2
                x_center = round(coords["x"], 6)
                y_center = round(coords["y"], 6)
                box_width = coords["width"]
                box_height = coords["height"]

                # 轉換為 YOLO 格式（相對值，並四捨五入）
                # x_center = round(coords["x"] / img_width, 6)
                # y_center = round(coords["y"] / img_height, 6)
                # box_width = round(coords["width"] / img_width, 6)
                # box_height = round(coords["height"] / img_height, 6)

                x_center = float(x_center / img_width)
                y_center = float(y_center / img_height)
                box_width = float(coords["width"] / img_width)
                box_height = float(coords["height"] / img_height)

                # line = f"{class_id} {x_center} {y_center} {box_width} {box_height}"
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                yolo_lines.append(line)

            with open(txt_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(yolo_lines))

                print(f"✅ 完成轉換（模擬 labelImg）：{txt_filename}")
                total_conversions += 1  # 每成功轉換一個檔案，計數器加一

# 在所有檔案處理完畢後，顯示總共轉換的數量
print(f"\n--- 轉換摘要 ---")
print(f"總共轉換了 {total_conversions} 個檔案。")
