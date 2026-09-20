<!-- 侧边栏 - 标签 -->
<!-- context="home"：全局热门标签，点击在首页原地筛选 -->
<!-- context="post"：与当前文章相关的标签（按共现排序），点击进标签页 -->
<template>
  <div v-if="tags.length" class="tags-cloud s-card">
    <div class="title">
      <i class="iconfont icon-hashtag"></i>
      <span class="title-name">{{ title }}</span>
    </div>
    <div class="all-tags">
      <a
        v-for="item in tags"
        :key="item.name"
        :href="item.href"
        class="tags"
        :title="item.title"
        @click="onTagClick($event, item)"
      >
        <span class="name">{{ item.name }}</span>
        <span class="num">{{ item.count }}</span>
      </a>
    </div>
  </div>
</template>

<script setup>
import { resolveCurrentPostId } from "@/utils/postUrl.mjs";
import { applyArchiveFilter } from "@/composables/useArchiveFilter";

const { theme, page, frontmatter } = useData();

const props = defineProps({
  // home = 全局热门；post = 与当前文章相关的标签
  context: {
    type: String,
    default: "home",
  },
});

const LIMIT_FALLBACK = 10;

const limit = computed(() => {
  const configured = Number(theme.value?.aside?.tags?.limit);
  return Number.isFinite(configured) && configured > 0 ? configured : LIMIT_FALLBACK;
});

const isPost = computed(() => props.context === "post");

const tagsData = computed(() => theme.value?.tagsData ?? {});

const globalCount = (name) => Number(tagsData.value[name]?.count) || 0;

// 全局热门：按文章数降序（原来直接按对象插入顺序渲染，"热门"名不副实）
const hotTags = computed(() =>
  Object.keys(tagsData.value)
    .map((name) => ({ name, count: globalCount(name) }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name)),
);

// 当前文章
const currentPost = computed(() => {
  const id = resolveCurrentPostId(page.value, frontmatter.value);
  if (id === undefined || id === null) return null;
  return theme.value?.postData?.find((post) => post.id === id) ?? null;
});

// 相关标签：统计其它文章与当前文章共现的标签，按共现次数排序
const relatedTags = computed(() => {
  const post = currentPost.value;
  if (!post) return [];

  const ownTags = new Set((post.tags || []).map(String));
  if (ownTags.size === 0) return [];

  const scores = new Map();
  (theme.value?.postData ?? []).forEach((other) => {
    if (!other || other.id === post.id) return;
    const otherTags = (other.tags || []).map(String);
    if (!otherTags.some((tag) => ownTags.has(tag))) return;
    otherTags.forEach((tag) => {
      if (ownTags.has(tag)) return;
      scores.set(tag, (scores.get(tag) || 0) + 1);
    });
  });

  return [...scores.entries()]
    .map(([name, score]) => ({ name, score, count: globalCount(name) }))
    .sort((a, b) => b.score - a.score || b.count - a.count || a.name.localeCompare(b.name))
    .slice(0, limit.value);
});

const tags = computed(() => {
  const list = isPost.value ? relatedTags.value : hotTags.value;
  // 文章没有相关标签时退回全局热门，别让侧边栏空着
  const source = list.length > 0 ? list : hotTags.value;

  return source.slice(0, limit.value).map((item) => ({
    name: item.name,
    count: item.count,
    title: `${item.name}（${item.count} 篇文章）`,
    // 首页原地筛选；文章页没有列表，进标签归档页
    href: isPost.value
      ? `/pages/tags/${encodeURIComponent(item.name)}`
      : `/?tag=${encodeURIComponent(item.name)}`,
  }));
});

const title = computed(() => {
  if (!isPost.value) return "热门标签";
  return relatedTags.value.length > 0 ? "相关标签" : "热门标签";
});

// 首页点标签：原地筛选，不跳转（复用历史文章那份状态）
const onTagClick = (event, item) => {
  if (isPost.value) return;
  // 列表不在当前页（例如普通页面）时交给浏览器正常跳转
  if (typeof document === "undefined" || !document.querySelector(".home-archive")) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
    return;
  }
  event.preventDefault();
  applyArchiveFilter({ tag: item.name, page: 1 });
  window.scrollTo({ top: 0, behavior: "smooth" });
};
</script>

<style lang="scss" scoped>
.tags-cloud {
  .all-tags {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px;

    .tags {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      max-width: 100%;
      min-width: 0;
      padding: 3px 9px;
      border-radius: 999px;
      background-color: var(--main-card-second-background);
      transition:
        color 0.25s,
        background-color 0.25s;

      .name {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 0.8rem;
      }

      // 计数做成小胶囊，比原来的上标更容易读
      .num {
        flex: 0 0 auto;
        min-width: 16px;
        padding: 0 4px;
        border-radius: 999px;
        font-size: 0.68rem;
        line-height: 1.5;
        text-align: center;
        color: var(--main-font-second-color);
        background-color: var(--main-card-background);
        transition:
          color 0.25s,
          background-color 0.25s;
      }

      &:hover {
        color: var(--main-card-background);
        background-color: var(--main-color);

        .num {
          color: var(--main-color);
          background-color: var(--main-card-background);
        }
      }
    }
  }
}
</style>
