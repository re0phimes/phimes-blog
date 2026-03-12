import test from "node:test";
import assert from "node:assert/strict";

import { transformSitemapItems } from "../.vitepress/theme/utils/sitemap.mjs";

const postData = [
  {
    title: "KV Cache（二）",
    date: new Date("2026-01-16").getTime(),
    permalink: "/posts/2026/01/16/transformer-kv-cache-attention-roofline",
    regularPath: "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.html",
    legacyCleanPath: "/posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。",
  },
];

test("transformSitemapItems excludes internal docs, reports, and duplicate pagination roots", () => {
  const items = [
    { url: "" },
    { url: "posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。" },
    { url: "docs/plans/2026-03-01-url-sitemap-implementation-plan" },
    { url: "plan/2025-12-31_09-58-40-home-most-popular-recent" },
    { url: "TEST_REPORT" },
    { url: "page" },
    { url: "page/" },
    { url: "page/1" },
    { url: "page/2" },
    { url: "pages/categories/llm-principles" },
  ];

  const result = transformSitemapItems(items, postData, {
    now: new Date("2026-03-12T00:00:00.000Z"),
  });

  assert.deepEqual(
    result.map((item) => item.url),
    ["/", "posts/2026/01/16/transformer-kv-cache-attention-roofline", "page/2", "pages/categories/llm-principles"],
  );
});

test("transformSitemapItems applies article seo metadata after permalink normalization", () => {
  const items = [
    { url: "posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。" },
  ];

  const [result] = transformSitemapItems(items, postData, {
    now: new Date("2026-03-12T00:00:00.000Z"),
  });

  assert.equal(result.url, "posts/2026/01/16/transformer-kv-cache-attention-roofline");
  assert.equal(result.priority, 0.7);
  assert.equal(result.changefreq, "monthly");
});
