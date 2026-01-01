/**
 * Home sections selectors (SSR-safe, pure functions).
 *
 * Note: Do not access `window`/`document` in this module.
 */

/**
 * Select Most Popular posts (MVP: curated list; optional fallback ranking).
 *
 * @param {Array<object>} postData - theme.postData
 * @param {object} mostPopularConfig - theme.home.highlights.mostPopular
 * @param {object} [options]
 * @param {string[]} [options.excludeIds] - post ids to exclude (used for de-dup with other sections)
 * @returns {Array<object>}
 */
export const selectMostPopularPosts = (postData, mostPopularConfig = {}, options = {}) => {
  if (!Array.isArray(postData) || postData.length === 0) return [];

  const rawLimit = Number(mostPopularConfig?.limit);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 12;

  const curated = Array.isArray(mostPopularConfig?.curated) ? mostPopularConfig.curated : [];
  const excludeIds = new Set(
    (Array.isArray(options.excludeIds) ? options.excludeIds : []).map((id) => String(id)),
  );

  const byPath = new Map();
  postData.forEach((post) => {
    if (!post || typeof post.regularPath !== "string") return;
    byPath.set(post.regularPath, post);
  });

  const selected = [];
  const selectedIds = new Set();

  const tryAdd = (post) => {
    if (!post || post.id === undefined || post.id === null) return false;
    const idKey = String(post.id);
    if (excludeIds.has(idKey)) return false;
    if (selectedIds.has(idKey)) return false;
    selected.push(post);
    selectedIds.add(idKey);
    return true;
  };

  // 1) curated list first (stable order, ignore missing paths)
  for (const path of curated) {
    if (selected.length >= limit) break;
    if (typeof path !== "string" || !path) continue;
    tryAdd(byPath.get(path));
  }

  // 2) fallback: only select posts with popularRank (actual view counts)
  if (selected.length < limit) {
    const candidates = postData.filter(
      (post) =>
        post &&
        post.id !== undefined &&
        post.id !== null &&
        Number.isFinite(post.popularRank) && // 只选有浏览量的文章
        !excludeIds.has(String(post.id)) &&
        !selectedIds.has(String(post.id)),
    );

    candidates.sort((a, b) => {
      // 按 popularRank 降序（浏览量越高越靠前）
      const aRank = a.popularRank;
      const bRank = b.popularRank;
      if (aRank !== bRank) return bRank - aRank;

      // 浏览量相同时按日期降序
      const aDate = Number(a.date) || 0;
      const bDate = Number(b.date) || 0;
      return bDate - aDate;
    });

    for (const post of candidates) {
      if (selected.length >= limit) break;
      tryAdd(post);
    }
  }

  // 3) 如果还不够，随机选择文章填充
  if (selected.length < limit) {
    const remaining = postData.filter(
      (post) =>
        post &&
        post.id !== undefined &&
        post.id !== null &&
        !excludeIds.has(String(post.id)) &&
        !selectedIds.has(String(post.id)),
    );

    // 简单随机打乱（使用日期作为伪随机种子，保证 SSR 一致性）
    const seed = new Date().getDate();
    remaining.sort((a, b) => {
      const aHash = (a.id * seed) % 100;
      const bHash = (b.id * seed) % 100;
      return aHash - bHash;
    });

    for (const post of remaining) {
      if (selected.length >= limit) break;
      tryAdd(post);
    }
  }

  return selected.slice(0, limit);
};

/**
 * Select Recent Posts (date desc; stable tie-breakers; supports excludeIds).
 *
 * @param {Array<object>} postData - theme.postData
 * @param {object} recentConfig - theme.home.highlights.recentPosts
 * @param {object} [options]
 * @param {Array<string|number>} [options.excludeIds]
 * @returns {Array<object>}
 */
export const selectRecentPosts = (postData, recentConfig = {}, options = {}) => {
  if (!Array.isArray(postData) || postData.length === 0) return [];

  const rawLimit = Number(recentConfig?.limit);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 6;

  const excludeIds = new Set(
    (Array.isArray(options.excludeIds) ? options.excludeIds : []).map((id) => String(id)),
  );

  const candidates = postData.filter((post) => post && post.id !== undefined && post.id !== null);
  candidates.sort((a, b) => {
    const aDate = Number(a.date) || 0;
    const bDate = Number(b.date) || 0;
    if (aDate !== bDate) return bDate - aDate;

    const aModified = Number(a.lastModified) || 0;
    const bModified = Number(b.lastModified) || 0;
    if (aModified !== bModified) return bModified - aModified;

    const aPath = typeof a.regularPath === "string" ? a.regularPath : "";
    const bPath = typeof b.regularPath === "string" ? b.regularPath : "";
    return aPath.localeCompare(bPath);
  });

  const selected = [];
  const selectedIds = new Set();
  for (const post of candidates) {
    if (selected.length >= limit) break;
    const idKey = String(post.id);
    if (excludeIds.has(idKey) || selectedIds.has(idKey)) continue;
    selected.push(post);
    selectedIds.add(idKey);
  }

  return selected.slice(0, limit);
};

/**
 * Select homepage Highlights sections.
 *
 * @param {Array<object>} postData - theme.postData
 * @param {object} themeConfig - theme config (expects themeConfig.home.highlights.*)
 * @returns {{ mostPopularPosts: Array<object>, recentPosts: Array<object> }}
 */
export const selectHomeHighlights = (postData, themeConfig = {}) => {
  const highlights = themeConfig?.home?.highlights;
  if (!highlights?.enable) {
    return { mostPopularPosts: [], recentPosts: [] };
  }

  const mostPopularPosts = selectMostPopularPosts(postData, highlights.mostPopular);
  const recentPosts = selectRecentPosts(postData, highlights.recentPosts, {
    excludeIds: mostPopularPosts.map((post) => post.id),
  });

  return { mostPopularPosts, recentPosts };
};
