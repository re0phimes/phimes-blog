#!/usr/bin/env node
/**
 * 扫描 posts 里引用的图片，读取尺寸生成 .vitepress/theme/assets/imageSize.mjs。
 *
 * 用途：给 markdown 里的 <img> 输出 width/height，避免图片加载完成时页面跳动（CLS）。
 * 图片不在清单里也不会出错，只是不输出尺寸属性。
 *
 * 用法：node scripts/generate-image-size.mjs
 * 依赖：sharp（已装为 devDependency，用来读取尺寸，也能顺带转 webp）
 */
import fs from "node:fs";
import path from "node:path";
import { globby } from "globby";

const ROOT = process.cwd();
const OUT = path.join(ROOT, ".vitepress/theme/assets/imageSize.mjs");

let sharp;
try {
  sharp = (await import("sharp")).default;
} catch {
  console.error("需要 sharp：npm i -D sharp");
  process.exit(1);
}

const files = await globby(["posts/**/*.md"]);
const urls = new Set();
for (const f of files) {
  const text = fs.readFileSync(f, "utf8");
  for (const m of text.matchAll(/!\[[^\]]*\]\(\s*(https?:\/\/[^\s)]+?)\s*(?:"[^"]*")?\s*\)/g)) {
    urls.add(m[1]);
  }
  const cover = text.match(/^cover:\s*(\S+)/m);
  if (cover && cover[1].startsWith("http")) urls.add(cover[1]);
}

const list = [...urls].sort();
console.log("待解析尺寸:", list.length);

const out = {};
let ok = 0;
for (const url of list) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const meta = await sharp(Buffer.from(await res.arrayBuffer())).metadata();
    if (meta.width && meta.height) {
      out[url] = [meta.width, meta.height];
      ok += 1;
    }
  } catch (e) {
    console.warn(`  ✗ ${url} -> ${e.message}`);
  }
}

const body = `// 图片尺寸清单（构建期生成，用于给 <img> 输出 width/height，避免加载时布局抖动）
// 由 scripts/generate-image-size.mjs 生成；新图不在表里也没关系，只是不输出尺寸。
export default ${JSON.stringify(out, null, 2)};
`;
fs.writeFileSync(OUT, body);
console.log(`写入 ${path.relative(ROOT, OUT)}：${ok}/${list.length} 条`);
