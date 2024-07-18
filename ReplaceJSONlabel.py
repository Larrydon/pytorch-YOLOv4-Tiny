import fnmatch
import sys
import os

from numpy import imag




if len(sys.argv) < 4:
    print("Usage: python ReplaceJSONlabel.py <lable JSON file path> <find TEXT> <replace TEXT>")
    sys.exit(1)

print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])
labeled_json_file_path = sys.argv[1]
find_text = sys.argv[2]
replace_text = sys.argv[3]


print("labeled_json_file_path=" + labeled_json_file_path)
print("find TEXT=" + find_text)
print("replace TEXT=" + replace_text)


def find_JSONs(source):
    # 用於匹配的圖檔擴展名
    # patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    patterns = ['*.json']
    
    # 用於存儲結果的清單
    image_files = []

    # 遍歷目錄樹
    for root, dirs, files in os.walk(source):
        # 檢查每個文件是否匹配圖檔擴展名
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                # 如果匹配，則加入清單
                image_files.append(os.path.join(root, filename))
            
    
    return image_files


# 找出所有圖檔
jsonfiles = find_JSONs(labeled_json_file_path)

# ## 輸出結果
# for jsfile in jsonfiles:
#     print(jsfile)

if len(jsonfiles) == 0:
    print("Can't find any json file by labeled_json_file_path")
    sys.exit(1)


try:
    i = 0    
    for jsfile in jsonfiles:
        i += 1
        with open(jsfile, 'r', encoding='utf-8') as file:
            content = file.read()
        # 替換所有的 LPD 為 LPR
        modified_content = content.replace(find_text, replace_text)
        
        # 將修改後的數據寫回 JSON 文件
        with open(jsfile, 'w', encoding='utf-8') as file:
            file.write(modified_content)
            #print("替換完成並寫入新的 JSON 文件。")        

    if i > 0:
        print("Finish replace...")
except Exception as e:
    print("Error:", e)
    
    

