const INTERNAL_PREFIXES = ["docs/plans/", "plan/"];
const EXCLUDED_EXACT_URLS = new Set(["TEST_REPORT", "page", "page/", "page/1", "pages/"]);

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
  return INTERNAL_PREFIXES.some((prefix) => url.startsWith(prefix));
};

export const transformSitemapItems = (items, postData, options = {}) => {
  const transformedCount = { value: 0 };
  const now = options.now ? new Date(options.now) : new Date();
  const threeMonthsAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);

  return items
    .map((rawItem) => {
      const normalizedUrl = normalizeInputUrl(rawItem.url);

      if (shouldExcludeUrl(normalizedUrl)) return null;

      if (normalizedUrl === "/") {
        return { ...rawItem, url: "/", priority: 1.0, changefreq: "weekly" };
      }

      if (normalizedUrl.startsWith("posts/")) {
        const post = findPostBySitemapUrl(normalizedUrl, postData);

        if (post && post.date) {
          const publicUrl = post.permalink?.replace(/^\//, "") || normalizedUrl;
          if (publicUrl !== normalizedUrl) transformedCount.value++;

          if (post.popular || post.popularRank) {
            return { ...rawItem, url: publicUrl, priority: 0.8, changefreq: "monthly" };
          }

          const postDate = new Date(post.date);
          if (postDate > threeMonthsAgo) {
            return { ...rawItem, url: publicUrl, priority: 0.7, changefreq: "monthly" };
          }

          return { ...rawItem, url: publicUrl, priority: 0.6, changefreq: "yearly" };
        }

        return { ...rawItem, url: normalizedUrl, priority: 0.5, changefreq: "yearly" };
      }

      if (normalizedUrl.startsWith("pages/categories/")) {
        return { ...rawItem, url: normalizedUrl, priority: 0.5, changefreq: "weekly" };
      }

      if (normalizedUrl.startsWith("pages/tags/")) {
        return { ...rawItem, url: normalizedUrl, priority: 0.5, changefreq: "weekly" };
      }

      if (normalizedUrl.includes("/archives")) {
        return { ...rawItem, url: normalizedUrl, priority: 0.4, changefreq: "monthly" };
      }

      return { ...rawItem, url: normalizedUrl, priority: 0.5, changefreq: "yearly" };
    })
    .filter(Boolean);
};
