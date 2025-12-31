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

const timeoutMs = Number(process.env.POPULAR_FETCH_TIMEOUT_MS || 8000);
const controller = new AbortController();
const timer = setTimeout(() => controller.abort(), timeoutMs);

let data;
try {
  const res = await fetch(url, { signal: controller.signal });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`);
  }
  data = await res.json();
} catch (error) {
  console.error(`[popular-cache] fetch failed: ${url}`, error);
  process.exit(1);
} finally {
  clearTimeout(timer);
}

if (!Array.isArray(data) && !(data && typeof data === "object")) {
  console.error("[popular-cache] invalid JSON shape: expected array or object");
  process.exit(1);
}

const absOutput = path.resolve(process.cwd(), outputFile);
await fs.ensureDir(path.dirname(absOutput));
await fs.writeJSON(absOutput, data, { spaces: 2 });

console.log(`[popular-cache] wrote: ${outputFile}`);

