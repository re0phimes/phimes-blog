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
    [
      "/",
      "posts/2026/01/16/transformer-kv-cache-attention-roofline",
      "pages/categories/llm-principles",
    ],
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
  // 文章日期 2026-01-16 落在 now 往前三个月内，按「较新文章」处理
  assert.equal(result.priority, 0.8);
  assert.equal(result.changefreq, "monthly");
});

test("independent pages drop VitePress's mtime-based lastmod", () => {
  // VitePress 会给每个页面塞一个基于文件 mtime 的 lastmod。
  // CI 的 actions/checkout 不保留 mtime，那个值等于构建时间 ——
  // 每次部署全站 lastmod 都刷新，爬虫于是以为所有页面都改过，白跑一遍重抓。
  const items = [
    { url: "pages/about", lastmod: "2022-01-01T00:00:00.000Z" },
    { url: "pages/privacy", lastmod: "2022-01-01T00:00:00.000Z" },
  ];

  const result = transformSitemapItems(items, postData, {
    now: new Date("2026-03-12T00:00:00.000Z"),
  });

  assert.equal(result.length, 2);
  for (const item of result) {
    assert.equal(item.lastmod, undefined, `${item.url} 不该带 lastmod`);
  }
});

test("siteLastmod, when provided, wins over the dropped mtime value", () => {
  const items = [{ url: "pages/about", lastmod: "2022-01-01T00:00:00.000Z" }];

  const [result] = transformSitemapItems(items, postData, {
    now: new Date("2026-03-12T00:00:00.000Z"),
    siteLastmod: "2025-06-01",
  });

  assert.equal(result.lastmod, new Date("2025-06-01").toISOString());
});

test("post lastmod comes from lastUpdated, not the mtime-ish lastModified", () => {
  const posts = [
    {
      ...postData[0],
      lastUpdated: new Date("2026-03-01").getTime(),
      // 这个值过去就是文件 mtime，在 CI 上等于构建时间，不该被 sitemap 采用
      lastModified: new Date("2026-03-10").getTime(),
    },
  ];
  const items = [
    { url: "posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。" },
  ];

  const [result] = transformSitemapItems(items, posts, {
    now: new Date("2026-03-12T00:00:00.000Z"),
  });

  assert.equal(result.lastmod, new Date("2026-03-01").toISOString());
});
