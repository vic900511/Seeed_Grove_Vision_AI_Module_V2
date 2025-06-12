import re

with open('test.cc', 'r') as f:
    content = f.read()

# 只抓 unsigned char 陣列的內容
matches = re.findall(r'\{([^}]*)\}', content)
if not matches:
    raise RuntimeError("No array data found")

data_str = matches[0]  # 取第一組大括號內的字串

# 抓出所有 0x?? 16 進位字串
bytes_hex = re.findall(r'0x([0-9a-fA-F]{2})', data_str)

# 讀長度定義（可選）
length_match = re.search(r'unsigned int\s+\w+_len\s*=\s*(\d+);', content)
length = int(length_match.group(1)) if length_match else None

if length:
    bytes_hex = bytes_hex[:length]

with open('model_back.tflite', 'wb') as fout:
    for hb in bytes_hex:
        fout.write(bytes([int(hb, 16)]))