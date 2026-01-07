# 自动修复并格式化
ruff check --fix .
ruff format .

# 查看还剩哪些问题需要手动修复
ruff check .

# 添加并提交
git add .
git commit -m "test generate 18 assets automatically"
