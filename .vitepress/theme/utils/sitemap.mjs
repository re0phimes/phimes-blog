/**
 * sitemap 生成策略
 *
 * 目标：让爬虫把抓取预算花在有独立价值的页面上，并且能判断「什么时候该重抓」。
 *
 * 处理的问题：
 *  1. /page/N 是重定向空壳（跳到 /?p=N），不该被收录 —— 白白消耗抓取预算
 *  2. 标签页 / 分类页原本没有 lastmod，爬虫只能靠猜
 *  3. 文章封面没有用 image sitemap 声明，进不了图片搜索
 *  4. 所有页面 priority 一刀切，不反映内容价值
 */

const INTERNAL_PREFIXES = ["docs/", "plan/"];
// 内部文档即使被误构建，也不该进 sitemap 主动喂给爬虫（srcExclude 是第一道防线）
const EXCLUDED_EXACT_URLS = new Set([
  "PROJECT_STATE",
  "TEST_REPORT",
  "page",
  "page/",
  "page/1",
  "pages/",
]);
// /page/N 是跳转到 /?p=N 的重定向页，页面本身没有内容
const EXCLUDED_PATTERNS = [/^page\/\d+$/, /^page\/\d+\//];

const normalizeInputUrl = (url) => {
  if (url === "" || url === "/") return "/";
  return String(url || "").replace(/^\//, "");
};

const findPostBySitemapUrl = (url, postData) =>
  postData.find((post) => {
    const candidates = new Set(
      [
        post?.permalink?.replace(/^\//, ""),
        post?.legacyCleanPath?.replace(/^\//, ""),
        post?.regularPath?.replace(/^\//, "").replace(/\.html$/, ""),
      ].filter(Boolean),
    );
    return candidates.has(url);
  });

const shouldExcludeUrl = (url) => {
  if (!url) return false;
  if (EXCLUDED_EXACT_URLS.has(url)) return true;
  if (INTERNAL_PREFIXES.some((prefix) => url.startsWith(prefix))) return true;
  if (EXCLUDED_PATTERNS.some((re) => re.test(url))) return true;
  return false;
};

const toDate = (value) => {
  if (!value) return null;
  const d = new Date(value);
  return Number.isFinite(d.getTime()) ? d : null;
};

/** 一组文章里最新的修改时间 */
const latestOf = (posts) => {
  const times = posts
    .map((p) => toDate(p.lastUpdated) || toDate(p.date))
    .filter(Boolean)
    .map((d) => d.getTime());
  return times.length ? new Date(Math.max(...times)) : null;
};

/** 把 sitemap 里的标签/分类 URL 还原成标签名 */
const decodeTaxonomyName = (url, kind) => {
  const prefix = `pages/${kind}/`;
  if (!url.startsWith(prefix)) return null;
  try {
    return decodeURIComponent(url.slice(prefix.length));
  } catch {
    return url.slice(prefix.length);
  }
};

export const transformSitemapItems = (items, postData, options = {}) => {
  const transformedCount = { value: 0 };
  const now = options.now ? new Date(options.now) : new Date();
  const threeMonthsAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
  const site = String(options.site || "").replace(/\/$/, "");

  const withLastmod = (item, date) => (date ? { ...item, lastmod: date.toISOString() } : item);

  return (
    items
      .map((rawItem) => {
        const normalizedUrl = normalizeInputUrl(rawItem.url);

        if (shouldExcludeUrl(normalizedUrl)) return null;

        // 首页
        if (normalizedUrl === "/") {
          return {
            ...rawItem,
            url: "/",
            priority: 1.0,
            changefreq: "daily",
            lastmod: latestOf(postData)?.toISOString() || now.toISOString(),
          };
        }

        // 文章
        if (normalizedUrl.startsWith("posts/")) {
          const post = findPostBySitemapUrl(normalizedUrl, postData);

          if (post && post.date) {
            const publicUrl = post.permalink?.replace(/^\//, "") || normalizedUrl;
            if (publicUrl !== normalizedUrl) transformedCount.value++;

            // 封面用 image sitemap 声明，才有机会进图片搜索
            const img =
              post.cover && site
                ? [
                    {
                      url: post.cover.startsWith("http") ? post.cover : `${site}${post.cover}`,
                    },
                  ]
                : undefined;

            const base = { ...rawItem, url: publicUrl, ...(img ? { img } : {}) };
            const lastmod =
              toDate(post.lastUpdated) || toDate(post.lastModified) || toDate(post.date);

            if (post.popular || post.popularRank) {
              return { ...withLastmod(base, lastmod), priority: 0.9, changefreq: "weekly" };
            }
            const postDate = toDate(post.date);
            if (postDate && postDate > threeMonthsAgo) {
              return { ...withLastmod(base, lastmod), priority: 0.8, changefreq: "monthly" };
            }
            return { ...withLastmod(base, lastmod), priority: 0.7, changefreq: "yearly" };
          }

          return { ...rawItem, url: normalizedUrl, priority: 0.5, changefreq: "yearly" };
        }

        // 分类页：lastmod 取该分类下最新文章
        const category = decodeTaxonomyName(normalizedUrl, "categories");
        if (category !== null) {
          const posts = postData.filter((p) => (p.categories || []).includes(category));
          const lastmod = latestOf(posts);
          return {
            ...withLastmod(rawItem, lastmod),
            url: normalizedUrl,
            priority: 0.6,
            changefreq: "weekly",
          };
        }

        // 标签页：lastmod 取该标签下最新文章；
        // 只有 1 篇文章的标签页属于薄内容，降低优先级（但保留 —— 它承担内链作用）
        const tag = decodeTaxonomyName(normalizedUrl, "tags");
        if (tag !== null) {
          const posts = postData.filter((p) => (p.tags || []).includes(tag));
          const lastmod = latestOf(posts);
          return {
            ...withLastmod(rawItem, lastmod),
            url: normalizedUrl,
            priority: posts.length >= 3 ? 0.6 : posts.length === 2 ? 0.5 : 0.3,
            changefreq: "weekly",
          };
        }

        if (normalizedUrl.includes("/archives")) {
          return {
            ...withLastmod(rawItem, latestOf(postData)),
            url: normalizedUrl,
            priority: 0.4,
            changefreq: "monthly",
          };
        }

        // 其余独立页面（about / link / project / privacy / cc …）
        return {
          ...withLastmod(rawItem, toDate(options.siteLastmod) || now),
          url: normalizedUrl,
          priority: 0.5,
          changefreq: "monthly",
        };
      })
      // rawItem.lastmod 有可能是字符串，统一成 ISO
      .map((item) => {
        if (item && item.lastmod && typeof item.lastmod === "string") {
          const d = toDate(item.lastmod);
          return d ? { ...item, lastmod: d.toISOString() } : item;
        }
        return item;
      })
      .filter(Boolean)
  );
};
