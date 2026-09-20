#!/usr/bin/env node
/**
 * Fetch and write a Most Popular cache file for build-time consumption.
 *
 * Usage:
 *   node scripts/fetch-popular-cache.mjs <url> [outputFile]
 *
 * Expected response JSON can be one of:
 * - ["/posts/a.html", "/posts/b.html"]                      // order list
 * - [{"regularPath":"/posts/a.html","score":123}, ...]      // score list
 * - {"/posts/a.html": 123, "/posts/b.html": 80}             // score map
 */

import fs from "fs-extra";
import path from "node:path";

const url = process.argv[2];
const outputFile = process.argv[3] || "data/popular.json";

if (!url) {
  console.error("Missing <url>.\nUsage: node scripts/fetch-popular-cache.mjs <url> [outputFile]");
  process.exit(1);
}

// 默认 8 秒实测不够：那个 /all 接口经常要 20~45 秒才返回，超时后脚本会把
// 空对象写回 data/popular.json，导致首页「按热度排序」静默失效。
const timeoutMs = Number(process.env.POPULAR_FETCH_TIMEOUT_MS || 45000);
const controller = new AbortController();
const timer = setTimeout(() => controller.abort(), timeoutMs);

const absOutput = path.resolve(process.cwd(), outputFile);

let data;
try {
  const res = await fetch(url, { signal: controller.signal });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`);
  }
  data = await res.json();
} catch (error) {
  console.warn(`[popular-cache] fetch failed (non-blocking): ${url}`, error.message);
  // 关键：拉取失败时不要覆盖已有缓存。
  // 之前这里写回 {}，把上一次成功抓到的浏览量全部抹掉了。
  if (await fs.pathExists(absOutput)) {
    console.warn(`[popular-cache] 保留已有缓存，不做覆盖: ${outputFile}`);
    process.exit(0);
  }
  console.warn(`[popular-cache] 没有已有缓存，写入空对象: ${outputFile}`);
  data = {};
} finally {
  clearTimeout(timer);
}

if (!Array.isArray(data) && !(data && typeof data === "object")) {
  console.error("[popular-cache] invalid JSON shape: expected array or object");
  process.exit(1);
}

await fs.ensureDir(path.dirname(absOutput));
await fs.writeJSON(absOutput, data, { spaces: 2 });

console.log(`[popular-cache] wrote: ${outputFile}`);
