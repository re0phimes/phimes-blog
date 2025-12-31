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
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 6;

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

  // 2) fallback (optional): popularRank / popular / top / date
  if (selected.length < limit) {
    const candidates = postData.filter(
      (post) =>
        post &&
        post.id !== undefined &&
        post.id !== null &&
        !excludeIds.has(String(post.id)) &&
        !selectedIds.has(String(post.id)),
    );

    candidates.sort((a, b) => {
      const aRank = Number.isFinite(a.popularRank) ? a.popularRank : -Infinity;
      const bRank = Number.isFinite(b.popularRank) ? b.popularRank : -Infinity;
      if (aRank !== bRank) return bRank - aRank;

      const aPopular = a.popular ? 1 : 0;
      const bPopular = b.popular ? 1 : 0;
      if (aPopular !== bPopular) return bPopular - aPopular;

      const aTop = a.top ? 1 : 0;
      const bTop = b.top ? 1 : 0;
      if (aTop !== bTop) return bTop - aTop;

      const aDate = Number(a.date) || 0;
      const bDate = Number(b.date) || 0;
      if (aDate !== bDate) return bDate - aDate;

      const aPath = typeof a.regularPath === "string" ? a.regularPath : "";
      const bPath = typeof b.regularPath === "string" ? b.regularPath : "";
      return aPath.localeCompare(bPath);
    });

    for (const post of candidates) {
      if (selected.length >= limit) break;
      tryAdd(post);
    }
  }

  return selected.slice(0, limit);
};
