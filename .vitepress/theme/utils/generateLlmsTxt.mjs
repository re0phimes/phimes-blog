import { writeFileSync } from "fs";
import fs from "fs-extra";
import path from "node:path";
import matter from "gray-matter";
import { getAllPosts } from "./getPostData.mjs";
import { getPostPublicPath } from "./postUrl.mjs";

/**
 * 生成 llms.txt（https://llmstxt.org/）
 *
 * 这是给大模型 / AI Agent 读的站点索引：纯 Markdown、结构固定、没有导航和广告噪音，
 * 比让模型去啃 HTML 更省 token，也更容易被正确引用。
 *
 * 两条设计决定：
 *   1. 文章主链接指向 `.md` 副本（由 generateMarkdownMirrors 生成），
 *      那是没有渲染噪音的纯正文；HTML 地址放在行末注释里备用。
 *   2. 摘要取 300 字，明显长于 HTML 的 meta description（那个受 SEO 长度约束，
 *      只有 150 字），因为模型的上下文比搜索引擎的结果页宽裕得多。
 */

/** 把 Markdown 正文压成一行纯文本，供摘要使用 */
const stripMarkdown = (text) =>
  String(text || "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, target, alias) => alias || target)
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*\|.*\|\s*$/gm, " ")
    .replace(/~~/g, "")
    .replace(/[*_`>#]/g, "")
    .replace(/\s+/g, " ")
    .trim();

/**
 * 取正文里最能代表全文的一段。
 * 跳过「前言」「TL;DR」「引言」这种只有几个字的标题行 —— 拿它当摘要开头没有信息量，
 * 而且几乎每篇都有，读起来千篇一律。全部段落都短才退回不筛选。
 */
const leadText = (content, minLength = 24) => {
  const blocks = String(content || "")
    .split(/\n{2,}/)
    .map((block) => stripMarkdown(block))
    .filter(Boolean);
  const substantial = blocks.filter((text) => text.length >= minLength);
  return (substantial.length ? substantial : blocks).join(" ");
};

const truncate = (text, max) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

/**
 * @param {*} config VitePress buildEnd config
 * @param {*} themeConfig 主题配置
 */
export const createLlmsTxt = async (config, themeConfig) => {
  const site = themeConfig.siteMeta.site.replace(/\/$/, "");
  const posts = (await getAllPosts()).sort((a, b) => new Date(b.date) - new Date(a.date));

  // 摘要：frontmatter 手写的优先，否则从正文现提取。
  // 不走 postData.description —— 那个已经在 getPostData 里截到 150 字给 SEO 用了。
  const buildSummary = async (post) => {
    let body = "";
    let authored = "";
    if (post.sourcePath) {
      try {
        const raw = await fs.readFile(path.resolve(post.sourcePath), "utf-8");
        const parsed = matter(raw);
        authored = parsed.data.description || parsed.data.summary || "";
        body = parsed.content;
      } catch {
        // 读不到就退回 postData 里的短描述，不因为摘要丢了整条索引
      }
    }
    const source = authored ? stripMarkdown(authored) : leadText(body);
    return truncate(source || stripMarkdown(post.description) || "", 300);
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
  lines.push(
    "- 正文纯文本：每篇文章的 URL 加上 `.md` 后缀，即为该文的原始 Markdown" +
      `（例：${site}/posts/2026/01/16/transformer-kv-cache-attention-roofline.md）。` +
      "下面「文章」一节列出的都是这个版本。",
  );
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
    const htmlUrl = `${site}${getPostPublicPath(post)}`;
    const date = post.date ? new Date(post.date).toISOString().slice(0, 10) : "";
    const desc = await buildSummary(post);
    const tags = (post.tags || []).join(", ");
    lines.push(`- [${post.title}](${htmlUrl}.md)${desc ? `: ${desc}` : ""}`);
    const metaParts = [date, tags].filter(Boolean).join(" · ");
    lines.push(`  <!-- ${metaParts}${metaParts ? " · " : ""}HTML: ${htmlUrl} -->`);
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
