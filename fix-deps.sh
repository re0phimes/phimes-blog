#!/bin/bash

echo "清理旧的依赖文件..."
rm -rf node_modules
rm -f package-lock.json

echo "使用npm重新安装依赖..."
npm install

echo "依赖安装完成，新的 package-lock.json 已生成"
echo "现在可以提交更改："
echo "git add ."
echo "git commit -m 'fix: 切换到npm并修复Vercel部署问题'"
echo "git push" 