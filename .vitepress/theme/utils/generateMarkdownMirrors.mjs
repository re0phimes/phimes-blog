import fs from "fs-extra";
import path from "node:path";
import matter from "gray-matter";
import { getAllPosts } from "./getPostData.mjs";
import { getPostPublicPath } from "./postUrl.mjs";
import { createWikilinkResolver } from "./markdownWikilink.mjs";

/**
 * 把源文里的 Obsidian 双链换成标准 Markdown 链接（绝对地址）
 * @returns {{ text: string, count: number }} 转换后的正文和成功转换的链接数
 */
const inlineWikilinks = (content, resolve, toAbsolute) => {
  let count = 0;
  const text = String(content).replace(/\[\[([^[\]\n]+)\]\]/g, (whole, inner) => {
    const info = resolve(inner);
    const label = info?.label || inner;
    const href = toAbsolute(info?.href);
    // 解析不到目标就退成纯文字，不制造死链
    if (!href) return label;
    count++;
    return `[${label}](${href})`;
  });
  return { text, count };
};

/**
 * 为每篇文章生成一份机器可读的 Markdown 副本
 *
 * 为什么要这个：
 *   站点正文是 VitePress 渲染的 HTML，一篇 20 万字节，里面塞满导航、侧栏、
 *   评论挂载点和 KaTeX 样式。搜索引擎不在乎，但 AI 爬虫（GPTBot / ClaudeBot /
 *   PerplexityBot）拿这些噪音很费 token，切片时也容易把页脚当正文。
 *
 *   所以构建期顺手把原始 Markdown 另存一份到 `{permalink}.md`，
 *   和 HTML 同路径、同 URL 结构，llms.txt 和 <link rel="alternate"> 都指向它。
 *   拿到的就是纯正文 + front matter 元数据，没有一行渲染噪音。
 *
 * 顺带做两件源文件里遗留的转换：
 *   1. Obsidian 双链 `[[笔记名]]` → 标准 Markdown 链接（绝对地址），
 *      解析规则和站点渲染时完全一致，用的是同一个 resolver。
 *   2. 站内绝对路径链接补全成完整 URL，脱离站点上下文也能点。
 *
 * @param {*} config VitePress buildEnd config（outDir 已解析为绝对路径）
 * @param {*} themeConfig 主题配置
 */
export const createMarkdownMirrors = async (config, themeConfig) => {
  const site = String(themeConfig.siteMeta?.site || "").replace(/\/$/, "");
  const posts = await getAllPosts();

  // 和渲染时同一套解析器，保证 [[双链]] 在 .md 里指向同一个地方
  const tags = [...new Set(posts.flatMap((post) => post.tags || []))];
  const resolveWikilink = createWikilinkResolver({ posts, tags });

  const toAbsolute = (href) => {
    if (!href) return null;
    if (/^https?:\/\//i.test(href)) return href;
    // 同页锚点保持相对，补上域名反而会指回文档开头
    if (href.startsWith("#")) return href;
    return `${site}${href.startsWith("/") ? "" : "/"}${href}`;
  };

  let written = 0;
  let wikilinkCount = 0;

  for (const post of posts) {
    const publicPath = getPostPublicPath(post);
    if (!post?.sourcePath || !publicPath) continue;

    let raw;
    try {
      raw = await fs.readFile(post.sourcePath, "utf-8");
    } catch (e) {
      console.warn(`[Markdown] 读取失败，跳过 ${post.sourcePath}: ${e.message}`);
      continue;
    }

    const { data, content } = matter(raw);

    // [[目标]] / [[目标|显示文字]] / [[目标#小节]]
    const { text: body, count } = inlineWikilinks(content, resolveWikilink, toAbsolute);
    wikilinkCount += count;

    const meta = {
      ...data,
      // 明确给出 HTML 版本，AI 引用时知道对应的可读页面在哪
      url: `${site}${publicPath}`,
    };

    const outPath = path.join(config.outDir, `${publicPath.replace(/^\//, "")}.md`);
    await fs.ensureDir(path.dirname(outPath));
    await fs.writeFile(outPath, matter.stringify(body.replace(/^\n+/, ""), meta), "utf-8");
    written++;
  }

  console.log(`[Markdown] 生成 ${written} 份 .md 副本，内联 ${wikilinkCount} 个双链`);
};

export default createMarkdownMirrors;
