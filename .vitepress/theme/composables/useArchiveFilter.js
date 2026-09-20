/**
 * 首页「历史文章」的筛选状态。
 *
 * 注意：不能用 VitePress 的 useRoute() —— 它不是 vue-router，
 * 而是自带的一个极简 router（`reactive(getDefaultRoute())`），
 * route 上没有 query 字段，读写 query 得走 window.location。
 *
 * 侧边栏的标签、列表里的筛选控件都改这一份状态，保证只有一处真相。
 */
import { reactive } from "vue";

const DEFAULTS = { sort: "time", tag: "", page: 1 };

export const archiveFilter = reactive({ ...DEFAULTS });

export const applyArchiveFilter = (patch = {}) => {
  Object.assign(archiveFilter, patch);
};

export const resetArchiveFilter = () => {
  Object.assign(archiveFilter, { ...DEFAULTS });
};

/** 从 URL 读取（页面加载时用） */
export const readArchiveFilterFromUrl = () => {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  archiveFilter.sort = params.get("sort") === "hot" ? "hot" : "time";
  archiveFilter.tag = params.get("tag") || "";
  const rawPage = Number(params.get("p"));
  archiveFilter.page = Number.isFinite(rawPage) && rawPage > 0 ? Math.floor(rawPage) : 1;
};

/** 写回 URL（用 replaceState，不产生历史记录，页面也不重新加载） */
export const writeArchiveFilterToUrl = () => {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams();
  if (archiveFilter.sort !== DEFAULTS.sort) params.set("sort", archiveFilter.sort);
  if (archiveFilter.tag) params.set("tag", archiveFilter.tag);
  if (archiveFilter.page > DEFAULTS.page) params.set("p", String(archiveFilter.page));

  const query = params.toString();
  const next = query ? `${window.location.pathname}?${query}` : window.location.pathname;
  if (next === `${window.location.pathname}${window.location.search}`) return;
  window.history.replaceState(null, "", next);
};
