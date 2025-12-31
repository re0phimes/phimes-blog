<!-- 首页 Highlights：Most Popular / Recent Posts -->
<template>
  <div v-if="enabled" class="home-highlights">
    <div class="highlights-grid">
      <section class="highlights-section s-card hover" aria-labelledby="home-highlights-popular">
        <div class="section-header">
          <span id="home-highlights-popular" class="section-title">
            {{ mostPopularTitle }}
          </span>
          <a v-if="mostPopularMoreLink" :href="mostPopularMoreLink" class="section-more">
            查看全部
            <i class="iconfont icon-arrow-right" />
          </a>
        </div>
        <div v-if="mostPopularPosts.length" class="section-list">
          <a
            v-for="(post, index) in mostPopularPosts"
            :key="post.id ?? `${post.regularPath}-${index}`"
            :href="post.regularPath"
            class="section-item"
          >
            <span class="rank">{{ index + 1 }}</span>
            <span class="title">{{ post.title }}</span>
            <span class="date">{{ formatTimestamp(post.date) }}</span>
          </a>
        </div>
        <div v-else class="empty">暂无内容</div>
      </section>

      <section class="highlights-section s-card hover" aria-labelledby="home-highlights-recent">
        <div class="section-header">
          <span id="home-highlights-recent" class="section-title">
            {{ recentPostsTitle }}
          </span>
          <a v-if="recentPostsMoreLink" :href="recentPostsMoreLink" class="section-more">
            查看全部
            <i class="iconfont icon-arrow-right" />
          </a>
        </div>
        <div v-if="recentPosts.length" class="section-list">
          <a
            v-for="(post, index) in recentPosts"
            :key="post.id ?? `${post.regularPath}-${index}`"
            :href="post.regularPath"
            class="section-item"
          >
            <span class="rank">{{ index + 1 }}</span>
            <span class="title">{{ post.title }}</span>
            <span class="date">{{ formatTimestamp(post.date) }}</span>
          </a>
        </div>
        <div v-else class="empty">暂无内容</div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";
import { formatTimestamp } from "@/utils/helper";
import { selectHomeHighlights } from "@/utils/homeSections.mjs";

const { theme } = useData();

const enabled = computed(() => Boolean(theme.value?.home?.highlights?.enable));

const mostPopularTitle = computed(
  () => theme.value?.home?.highlights?.mostPopular?.title || "Most Popular",
);
const mostPopularMoreLink = computed(() => theme.value?.home?.highlights?.mostPopular?.moreLink || "");

const recentPostsTitle = computed(() => theme.value?.home?.highlights?.recentPosts?.title || "Recent Posts");
const recentPostsMoreLink = computed(() => theme.value?.home?.highlights?.recentPosts?.moreLink || "");

const highlightsData = computed(() => selectHomeHighlights(theme.value?.postData, theme.value));
const mostPopularPosts = computed(() => highlightsData.value.mostPopularPosts ?? []);
const recentPosts = computed(() => highlightsData.value.recentPosts ?? []);
</script>

<style lang="scss" scoped>
.home-highlights {
  margin-bottom: 1rem;
  animation: fade-up 0.6s 0.2s backwards;
}

.highlights-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.highlights-section {
  cursor: default;
}

.section-header {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  gap: 12px;

  .section-title {
    font-size: 1.125rem;
    font-weight: bold;
    opacity: 0.9;
  }

  .section-more {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    font-size: 14px;
    opacity: 0.65;
    transition:
      opacity 0.3s,
      color 0.3s;

    .iconfont {
      margin-left: 6px;
      font-size: 0.875rem;
      opacity: 0.8;
      transition: color 0.3s;
    }

    &:hover {
      opacity: 1;
      color: var(--main-color);
      .iconfont {
        color: var(--main-color);
      }
    }
  }
}

.section-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.section-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--main-card-border);
  background-color: var(--main-card-second-background);
  transition:
    border-color 0.3s,
    box-shadow 0.3s,
    background-color 0.3s,
    color 0.3s;

  .rank {
    width: 24px;
    flex: 0 0 24px;
    font-weight: bold;
    opacity: 0.75;
    color: var(--main-color);
  }

  .title {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding-right: 10px;
  }

  .date {
    font-size: 13px;
    opacity: 0.6;
    white-space: nowrap;
  }

  &:hover {
    border-color: var(--main-color);
    box-shadow: 0 8px 16px -4px var(--main-color-bg);
    background-color: var(--main-color-bg);
  }

  &:hover {
    .title {
      color: var(--main-color);
    }
  }

  &:active {
    transform: scale(0.99);
  }
}

.empty {
  padding: 12px;
  font-size: 14px;
  opacity: 0.6;
  border-radius: 12px;
  border: 1px dashed var(--main-card-border);
}
</style>
