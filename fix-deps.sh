#!/bin/bash

echo "正在重新安装依赖..."
pnpm install

echo "依赖安装完成，新的 pnpm-lock.yaml 已生成"
echo "现在可以提交更改："
echo "git add ."
echo "git commit -m 'fix: 更新pnpm-lock.yaml以修复Vercel部署问题'"
echo "git push" 