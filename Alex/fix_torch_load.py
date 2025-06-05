# 这个脚本用于修复 Transformer_Fixed.py 中的 torch.load() 调用
# 添加 weights_only=True 参数以提高安全性

import re

# 读取文件内容
with open('Transformer_Fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 torch.load() 调用
pattern = r'(torch\.load\s*\([^,)]*)(\)|,)'
replacement = r'\1, weights_only=True\2'
new_content = re.sub(pattern, replacement, content)

# 写回文件
with open('Transformer_Fixed.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("已成功更新 torch.load() 调用，添加了 weights_only=True 参数。")
