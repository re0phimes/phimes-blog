/**
 * Obsidian 双链语法 [[...]] 支持
 *
 * 这些文章的源文件是 Obsidian 笔记，正文里大量使用 [[笔记名]] / [[笔记名#标题]]。
 * VitePress 默认不认识这套语法，会把 [[...]] 原样当文字渲染出来。
 *
 * 解析优先级：
 *   1. [[#标题]]            → 当前页锚点
 *   2. [[文件名]] / [[标题]] → 文章链接（文件名优先，与 Obsidian 一致）
 *   3. [[标签]]             → 标签归档页
 *   4. 以上都不匹配          → 只渲染可见文字，不产生死链
 *
 * Obsidian 的别名写法 [[目标|显示文字]] 同样支持。
 */

// VitePress 生成标题锚点用的 slugify（与 @mdit-vue/shared 保持一致）
const rControl = /[\u0000-\u001f]/g;
const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g;
const rCombining = /[\u0300-\u036f]/g;

const slugifyHeading = (input) =>
  String(input)
    .normalize("NFKD")
    .replace(rCombining, "")
    .replace(rControl, "")
    .replace(rSpecial, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/^(\d)/, "_$1")
    .toLowerCase();

/** 归一化用于匹配的笔记名：去掉 Obsidian 追加的日期、结尾标点，忽略大小写 */
const normalizeName = (input) =>
  String(input)
    .replace(/\s+\d{4}-\d{2}-\d{2}\s*$/, "")
    .trim()
    .replace(/[？?！!。．.：:]+$/, "")
    .toLowerCase();

const toBasename = (regularPath) => {
  const file = String(regularPath || "").split("/").pop() || "";
  return file.replace(/\.[^.]+$/, "");
};

/**
 * 建立 [[...]] → 链接 的解析器
 * @param {{ posts?: Array, tags?: Array }} data
 */
export const createWikilinkResolver = ({ posts = [], tags = [] } = {}) => {
  // 文件名与标题都能命中，文件名优先
  const byName = new Map();
  const byTitle = new Map();

  posts.forEach((post) => {
    if (!post?.permalink) return;
    const name = normalizeName(toBasename(post.regularPath));
    const title = normalizeName(post.title);
    if (name && !byName.has(name)) byName.set(name, post);
    if (title && !byTitle.has(title)) byTitle.set(title, post);
  });

  const tagLookup = new Map();
  tags.forEach((tag) => {
    if (tag) tagLookup.set(String(tag).toLowerCase(), String(tag));
  });

  const matchPost = (target) => {
    const key = normalizeName(target);
    if (!key) return null;
    if (byName.has(key)) return byName.get(key);
    if (byTitle.has(key)) return byTitle.get(key);

    // 允许「系列名前缀」「标题被截短」这类写法：
    // [[工程实现系列：从什么都不会到QLoRA分布式DPO（二）- wandb...]]
    // 对应文件「从什么都不会到QLoRA分布式DPO（二）- wandb...」
    if (key.length < 6) return null;
    for (const table of [byName, byTitle]) {
      for (const [candidate, post] of table) {
        if (candidate.length < 6) continue;
        if (key.startsWith(candidate) || key.endsWith(candidate)) return post;
      }
    }
    return null;
  };

  return (rawTarget, env = {}) => {
    const [targetPart, aliasPart] = String(rawTarget).split("|");
    const target = targetPart.trim();
    const hashIndex = target.indexOf("#");
    const pagePart = hashIndex === -1 ? target : target.slice(0, hashIndex);
    const heading = hashIndex === -1 ? "" : target.slice(hashIndex + 1).split("#").pop().trim();
    // 显示文字：优先别名，其次笔记名（不含 # 小节），最后退回小节名
    const label = (aliasPart ?? (pagePart || heading)).trim() || target;

    if (!pagePart) {
      // [[#标题]] —— 当前页锚点
      return heading
        ? { href: `#${slugifyHeading(heading)}`, label, title: heading }
        : { label };
    }

    const post = matchPost(pagePart);
    if (post) {
      const href = heading
        ? `${post.permalink}#${slugifyHeading(heading)}`
        : post.permalink;
      return { href, label, title: post.title || label };
    }

    // 兜底：当作标签引用
    const tag = tagLookup.get(pagePart.toLowerCase());
    if (tag) {
      return { href: `/pages/tags/${encodeURIComponent(tag)}`, label, title: `标签：${tag}` };
    }

    return { label };
  };
};

export const markdownWikilink = (md, resolver) => {
  md.inline.ruler.before("link", "obsidian_wikilink", (state, silent) => {
    const { pos } = state;
    if (state.src.charCodeAt(pos) !== 0x5b || state.src.charCodeAt(pos + 1) !== 0x5b) {
      return false;
    }

    const end = state.src.indexOf("]]", pos + 2);
    if (end === -1) return false;

    const inner = state.src.slice(pos + 2, end);
    if (!inner.trim() || inner.includes("\n") || inner.includes("[[")) return false;

    if (!silent) {
      const token = state.push("obsidian_wikilink", "", 0);
      token.content = inner;
      token.meta = resolver ? resolver(inner, state.env) : { label: inner };
    }

    state.pos = end + 2;
    return true;
  });

  md.renderer.rules.obsidian_wikilink = (tokens, idx) => {
    const token = tokens[idx];
    const info = token.meta || { label: token.content };
    const label = md.utils.escapeHtml(info.label || token.content);

    if (!info.href) return label;

    const title = info.title ? ` title="${md.utils.escapeHtml(info.title)}"` : "";
    return `<a class="wikilink" href="${md.utils.escapeHtml(info.href)}"${title}>${label}</a>`;
  };
};

export default markdownWikilink;
