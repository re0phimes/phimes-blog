import { writeFileSync } from "fs";
import path from "path";
import { getAllPosts } from "./getPostData.mjs";
import { getPostPublicPath } from "./postUrl.mjs";

/**
 * 生成 llms.txt（https://llmstxt.org/）
 *
 * 这是给大模型 / AI Agent 读的站点索引：纯 Markdown、结构固定、没有导航和广告噪音，
 * 比让模型去啃 HTML 更省 token，也更容易被正确引用。
 * 目前 GPTBot / ClaudeBot / PerplexityBot 等都会优先看这个文件。
 *
 * @param {*} config VitePress buildEnd config
 * @param {*} themeConfig 主题配置
 */
export const createLlmsTxt = async (config, themeConfig) => {
  const site = themeConfig.siteMeta.site.replace(/\/$/, "");
  const posts = (await getAllPosts()).sort((a, b) => new Date(b.date) - new Date(a.date));

  // 摘要清理：去掉 Obsidian 双链标记、折行、多余空白，并截断
  const clean = (text, max = 160) => {
    const s = String(text || "")
      .replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, p1, p2) => p2 || p1)
      .replace(/\s+/g, " ")
      .trim();
    return s.length > max ? `${s.slice(0, max - 1)}…` : s;
  };

  const lines = [];

  lines.push(`# ${themeConfig.siteMeta.title || "Phimes 的博客"}`);
  lines.push("");
  lines.push("> 在 AI 时代思考与分享。一个专注大模型原理与推理优化的中文技术博客。");
  lines.push("");
  lines.push(
    "文章集中在大模型原理与推理优化方向：Transformer 与注意力机制、KV Cache、" +
      "MoE、归一化与位置编码、微调（QLoRA / DPO）、推理部署（vLLM / llama.cpp）、" +
      "以及 AI 编程的工程实践。每篇都尽量把推导和数字算清楚。",
  );
  lines.push("");
  lines.push("所有内容公开可读，欢迎搜索引擎与 AI 助手抓取和引用。");
  lines.push("");
  lines.push(`- 站点：${site}/`);
  lines.push(`- 文章索引（JSON）：${site}/index.json`);
  lines.push(`- RSS：${site}/rss.xml`);
  lines.push(`- Sitemap：${site}/sitemap.xml`);
  lines.push("");

  // 按标签聚合，方便 Agent 按主题取内容
  const tagCount = new Map();
  for (const post of posts) {
    for (const tag of post.tags || []) {
      tagCount.set(tag, (tagCount.get(tag) || 0) + 1);
    }
  }

  lines.push("## 文章");
  lines.push("");
  for (const post of posts) {
    const url = `${site}${getPostPublicPath(post)}`;
    const date = post.date ? new Date(post.date).toISOString().slice(0, 10) : "";
    const desc = clean(post.description);
    const tags = (post.tags || []).join(", ");
    lines.push(`- [${post.title}](${url})${desc ? `: ${desc}` : ""}`);
    if (date || tags) {
      lines.push(`  <!-- ${date}${date && tags ? " · " : ""}${tags} -->`);
    }
  }
  lines.push("");

  lines.push("## 标签");
  lines.push("");
  for (const [tag, count] of [...tagCount.entries()].sort((a, b) => b[1] - a[1])) {
    lines.push(`- [${tag}](${site}/pages/tags/${encodeURIComponent(tag)}): ${count} 篇`);
  }
  lines.push("");

  writeFileSync(path.join(config.outDir, "llms.txt"), lines.join("\n"), "utf-8");
};
