<!-- 首页历史文章：客户端排序 / 标签筛选 / 客户端分页，状态写进 URL -->
<template>
  <section class="home-archive">
    <div class="archive-head">
      <div class="archive-title">
        历史文章
        <span class="archive-count">{{ filteredPosts.length }} 篇</span>
      </div>
      <!-- 分割线分隔的三段筛选 -->
      <div class="archive-filters" role="group" aria-label="文章筛选">
        <button
          type="button"
          class="filter-item"
          :class="{ active: sort === 'time' }"
          @click="setSort('time')"
        >
          时间
        </button>
        <button
          type="button"
          class="filter-item"
          :class="{ active: sort === 'hot' }"
          :disabled="!hasPopularity"
          :title="hasPopularity ? '按浏览量排序' : '暂无浏览量数据'"
          @click="setSort('hot')"
        >
          热度
        </button>
        <div class="filter-item tag-filter">
          <button
            type="button"
            class="tag-trigger"
            :class="{ active: !!activeTag }"
            @click.stop="toggleTagPanel"
          >
            标签<template v-if="activeTag"
              ><span class="tag-current">{{ activeTag }}</span></template
            >
          </button>
          <div v-if="tagPanelOpen" class="tag-panel" @click.stop>
            <button v-if="activeTag" type="button" class="tag-option clear" @click="setTag('')">
              清除筛选
            </button>
            <button
              v-for="item in tagOptions"
              :key="item.name"
              type="button"
              class="tag-option"
              :class="{ active: item.name === activeTag }"
              @click="setTag(item.name)"
            >
              <span class="tag-name">{{ item.name }}</span>
              <span class="tag-count">{{ item.count }}</span>
            </button>
          </div>
        </div>
      </div>
    </div>

    <PostList
      :listData="pagedPosts"
      :startIndex="(page - 1) * pageSize"
      :showPopular="sort === 'hot'"
    />

    <div v-if="!filteredPosts.length" class="archive-empty">
      <span>没有符合条件的文章</span>
      <button type="button" @click="resetFilters">清除筛选</button>
    </div>

    <div v-if="totalPages > 1" class="archive-pager">
      <button
        v-for="num in totalPages"
        :key="num"
        type="button"
        class="pager-btn"
        :class="{ active: num === page }"
        @click="setPage(num)"
      >
        {{ num }}
      </button>
    </div>
  </section>
</template>

<script setup>
import { selectHomeFeed } from "@/utils/homeSections.mjs";

const { theme } = useData();

import {
  archiveFilter,
  applyArchiveFilter,
  readArchiveFilterFromUrl,
  resetArchiveFilter,
  writeArchiveFilterToUrl,
} from "@/composables/useArchiveFilter";

// 只有这一份状态：侧边栏标签、列表筛选控件改的都是它
const sort = computed(() => archiveFilter.sort);
const activeTag = computed(() => archiveFilter.tag);
const page = computed(() => archiveFilter.page);

const tagPanelOpen = ref(false);

const pageSize = computed(() => Number(theme.value?.postSize) || 8);

// 历史文章 = 全部文章流去掉首屏的头条与封面次条
const feedPosts = computed(() => selectHomeFeed(theme.value?.postData, theme.value).feed);

// 有没有真实浏览量：data/popular.json 由 build 的 fetch-popular 步骤写入
const hasPopularity = computed(() =>
  feedPosts.value.some((post) => Number.isFinite(Number(post.popularRank))),
);

const tagOptions = computed(() => {
  const tagsData = theme.value?.tagsData ?? {};
  return Object.entries(tagsData)
    .map(([name, value]) => ({ name, count: Number(value?.count) || 0 }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
});

const filteredPosts = computed(() => {
  let posts = feedPosts.value;

  if (activeTag.value) {
    posts = posts.filter((post) =>
      (post.tags || []).some((tag) => String(tag) === activeTag.value),
    );
  }

  if (sort.value === "hot") {
    // 没有浏览量的排在后面，同热度再按时间倒序
    posts = [...posts].sort((a, b) => {
      const aRank = Number(a.popularRank);
      const bRank = Number(b.popularRank);
      const aHas = Number.isFinite(aRank);
      const bHas = Number.isFinite(bRank);
      if (aHas !== bHas) return aHas ? -1 : 1;
      if (aHas && bHas && aRank !== bRank) return bRank - aRank;
      return (Number(b.date) || 0) - (Number(a.date) || 0);
    });
  }

  return posts;
});

const totalPages = computed(() =>
  Math.max(1, Math.ceil(filteredPosts.value.length / pageSize.value)),
);

const pagedPosts = computed(() => {
  const start = (page.value - 1) * pageSize.value;
  return filteredPosts.value.slice(start, start + pageSize.value);
});

const setSort = (value) => {
  if (value === "hot" && !hasPopularity.value) return;
  applyArchiveFilter({ sort: value, page: 1 });
};

const setTag = (value) => {
  applyArchiveFilter({ tag: value === activeTag.value ? "" : value, page: 1 });
  tagPanelOpen.value = false;
};

const setPage = (value) => {
  applyArchiveFilter({ page: Math.min(Math.max(1, value), totalPages.value) });
  if (typeof window !== "undefined") {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
};

const resetFilters = () => {
  resetArchiveFilter();
};

const toggleTagPanel = () => {
  tagPanelOpen.value = !tagPanelOpen.value;
};

const closeTagPanel = () => {
  tagPanelOpen.value = false;
};

watch(archiveFilter, () => {
  // 筛选变化后页码可能越界
  if (page.value > totalPages.value) {
    archiveFilter.page = totalPages.value;
  }
  writeArchiveFilterToUrl();
});

onMounted(() => {
  // SSR 阶段拿不到 URL，只能在挂载后同步（也避免 hydration 不一致）
  readArchiveFilterFromUrl();
  document.addEventListener("click", closeTagPanel);
});

onBeforeUnmount(() => {
  document.removeEventListener("click", closeTagPanel);
});
</script>

<style lang="scss" scoped>
.home-archive {
  width: 100%;
  margin-top: 3.2rem;

  .archive-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--main-card-border);
    margin-bottom: 1.1rem;
  }

  .archive-title {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    font-family: var(--main-serif-family);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--main-font-color);

    .archive-count {
      font-family: var(--main-font-family);
      font-size: 0.78rem;
      font-weight: 400;
      opacity: 0.55;
    }
  }

  // 三个筛选项之间用竖线分隔
  .archive-filters {
    display: flex;
    align-items: center;
    font-size: 0.85rem;

    .filter-item {
      display: flex;
      align-items: center;
      padding: 0 0.85rem;
      background: none;
      border: none;
      border-left: 1px solid var(--main-card-border);
      font-family: var(--main-font-family);
      font-size: 0.85rem;
      color: var(--main-font-second-color);
      cursor: pointer;
      transition: color 0.3s;

      &:first-child {
        border-left: none;
      }

      &:hover:not(:disabled) {
        color: var(--main-color);
      }

      &.active {
        color: var(--main-color);
        font-weight: 600;
      }

      &:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
    }
  }

  .tag-filter {
    position: relative;

    .tag-trigger {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0;
      background: none;
      border: none;
      font-family: var(--main-font-family);
      font-size: 0.85rem;
      color: inherit;
      font-weight: inherit;
      cursor: pointer;

      .tag-current {
        padding: 1px 8px;
        border-radius: 999px;
        background-color: var(--main-color-bg);
        color: var(--main-color);
        font-size: 0.75rem;
      }
    }

    .tag-panel {
      position: absolute;
      top: calc(100% + 0.6rem);
      right: 0;
      z-index: 20;
      width: 220px;
      max-height: 320px;
      overflow-y: auto;
      padding: 0.4rem;
      border-radius: 12px;
      border: 1px solid var(--main-card-border);
      background-color: var(--main-card-background);
      box-shadow: 0 8px 24px -8px var(--main-border-shadow);

      .tag-option {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        width: 100%;
        padding: 0.45rem 0.6rem;
        border: none;
        border-radius: 8px;
        background: none;
        font-family: var(--main-font-family);
        font-size: 0.82rem;
        color: var(--main-font-second-color);
        text-align: left;
        cursor: pointer;
        transition:
          color 0.2s,
          background-color 0.2s;

        .tag-count {
          font-size: 0.72rem;
          opacity: 0.55;
        }

        &:hover {
          color: var(--main-color);
          background-color: var(--main-color-bg);
        }

        &.active {
          color: var(--main-color);
          background-color: var(--main-color-bg);
          font-weight: 600;
        }

        &.clear {
          justify-content: center;
          border-bottom: 1px solid var(--main-card-border);
          border-radius: 8px 8px 0 0;
          margin-bottom: 0.2rem;
        }
      }
    }
  }

  .archive-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.8rem;
    padding: 3rem 0;
    font-size: 0.9rem;
    color: var(--main-font-second-color);

    button {
      padding: 4px 12px;
      border-radius: 999px;
      border: 1px solid var(--main-card-border);
      background: none;
      font-family: var(--main-font-family);
      font-size: 0.82rem;
      color: var(--main-color);
      cursor: pointer;

      &:hover {
        background-color: var(--main-color-bg);
      }
    }
  }

  .archive-pager {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1.6rem;

    .pager-btn {
      min-width: 32px;
      height: 32px;
      padding: 0 0.5rem;
      border-radius: 8px;
      border: 1px solid var(--main-card-border);
      background-color: var(--main-card-background);
      font-family: var(--main-font-family);
      font-size: 0.85rem;
      color: var(--main-font-second-color);
      cursor: pointer;
      transition: all 0.25s;

      &:hover {
        color: var(--main-color);
        border-color: var(--main-color);
      }

      &.active {
        color: #fff;
        background-color: var(--main-color);
        border-color: var(--main-color);
      }
    }
  }

  @media (max-width: 768px) {
    margin-top: 2.2rem;

    .archive-head {
      align-items: flex-start;
      flex-direction: column;
      gap: 0.7rem;
    }

    .archive-filters .filter-item {
      &:first-child {
        padding-left: 0;
      }
    }
  }
}
</style>
