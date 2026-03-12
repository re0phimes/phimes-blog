import test from "node:test";
import assert from "node:assert/strict";

import {
  buildLegacyRedirectHtml,
  buildPageViewPathCandidates,
  buildPostRewriteRules,
  buildPostUrlData,
  resolveCurrentPostId,
} from "../.vitepress/theme/utils/postUrl.mjs";

test("buildPostUrlData prefers topic slug and keeps legacy html path", () => {
  const result = buildPostUrlData({
    relativePath: "posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.md",
    date: "2026-01-16",
    topic: ["transformer", "kv-cache", "attention", "roofline"],
    tags: ["llm", "KV-Cache", "MQA"],
  });

  assert.equal(result.permalink, "/posts/2026/01/16/transformer-kv-cache-attention-roofline");
  assert.equal(
    result.legacyPath,
    "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.html",
  );
  assert.equal(result.legacyCleanPath, "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。");
  assert.equal(result.id, "20260116-transformer-kv-cache-attention-roofline");
});

test("buildPostUrlData falls back to latin tags when topic is absent", () => {
  const result = buildPostUrlData({
    relativePath: "posts/2025/从tools-use谈Deepseek联网搜索.md",
    date: "2025-02-01",
    tags: ["llm", "Tools", "中文标签"],
  });

  assert.equal(result.permalink, "/posts/2025/02/01/llm-tools");
  assert.equal(result.slug, "llm-tools");
});

test("buildPageViewPathCandidates returns permalink and legacy fallbacks", () => {
  const candidates = buildPageViewPathCandidates({
    permalink: "/posts/2026/01/16/transformer-kv-cache-attention-roofline",
    legacyPath: "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.html",
  });

  assert.deepEqual(candidates, [
    "/posts/2026/01/16/transformer-kv-cache-attention-roofline",
    "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。",
    "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.html",
  ]);
});

test("buildPostRewriteRules maps legacy markdown sources to permalink routes", () => {
  const rewrites = buildPostRewriteRules([
    {
      legacyPath: "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.html",
      permalink: "/posts/2026/01/16/transformer-kv-cache-attention-roofline",
    },
  ]);

  assert.deepEqual(rewrites, {
    "posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.md":
      "posts/2026/01/16/transformer-kv-cache-attention-roofline.md",
  });
});

test("buildLegacyRedirectHtml points legacy pages to the canonical permalink", () => {
  const html = buildLegacyRedirectHtml("/posts/2026/01/16/transformer-kv-cache-attention-roofline");

  assert.match(html, /http-equiv="refresh" content="0; url=\/posts\/2026\/01\/16\/transformer-kv-cache-attention-roofline"/);
  assert.match(html, /location\.replace\("\/posts\/2026\/01\/16\/transformer-kv-cache-attention-roofline"\)/);
  assert.match(html, /rel="canonical" href="\/posts\/2026\/01\/16\/transformer-kv-cache-attention-roofline"/);
});

test("resolveCurrentPostId prefers frontmatter id for rewritten post routes", () => {
  const id = resolveCurrentPostId(
    {
      relativePath: "posts/2026/01/16/transformer-kv-cache-attention-roofline.md",
      filePath: "posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.md",
    },
    {
      id: "20260116-transformer-kv-cache-attention-roofline",
    },
  );

  assert.equal(id, "20260116-transformer-kv-cache-attention-roofline");
});
