import re
import os

# 读取setup.py文件
with open('setup.py', 'r') as f:
    lines = f.readlines()

# 找到版本号行
for i, line in enumerate(lines):
    match = re.search(r"version='(\d+)\.(\d+)\.(\d+)',", line)
    if match:
        major, minor, patch = map(int, match.groups())
        break
else:
    raise ValueError('Version number not found in setup.py')

# 更新版本号
patch += 1
if patch == 10:
    patch = 0
    minor += 1
    if minor == 10:
        minor = 0
        major += 1
lines[i] = f"    version='{major}.{minor}.{patch}',  # version\n"

# 将更新后的内容写回setup.py文件
with open('setup.py', 'w') as f:
    f.writelines(lines)

# 更新__init__.py文件中的__version__
init_file = 'specnn4pde/__init__.py'
with open(init_file, 'r') as f:
    init_lines = f.readlines()

for i, line in enumerate(init_lines):
    if line.startswith('__version__'):
        init_lines[i] = f"__version__ = '{major}.{minor}.{patch}'\n"
        break
else:
    # 如果没有找到__version__，则添加
    init_lines.append(f"__version__ = '{major}.{minor}.{patch}'\n")

with open(init_file, 'w') as f:
    f.writelines(init_lines)

# 清理旧的构建文件（避免错误的元数据）
if os.name == "posix":  # macOS / Linux
    os.system("rm -rf build dist specnn4pde.egg-info")
elif os.name == "nt":  # Windows
    os.system("rmdir /s /q build dist specnn4pde.egg-info")

# 添加所有更改到git
os.system('git add .')

# 提交更改
os.system(f'git commit -m "v{major}.{minor}.{patch}"')

# push更改到远程仓库
os.system('git push')

# 执行发布命令
os.system('python -m build')
os.system('twine upload dist/*')