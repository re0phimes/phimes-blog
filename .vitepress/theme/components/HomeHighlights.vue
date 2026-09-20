<!-- 首页头条下方的封面次条（最新文章） -->
<template>
  <section v-if="featuredPosts.length" class="home-featured">
    <div class="featured-head">
      <span class="featured-title">{{ title }}</span>
      <a v-if="moreLink" class="featured-more" :href="moreLink">更多 →</a>
    </div>
    <div class="featured-grid">
      <a
        v-for="post in featuredPosts"
        :key="post.id ?? post.permalink ?? post.regularPath"
        :href="getPostPublicPath(post)"
        class="featured-card s-card hover"
      >
        <div class="card-cover">
          <img
            v-if="post.cover"
            :src="post.cover"
            :alt="post.title"
            loading="lazy"
            @error="(e) => (e.target.style.display = 'none')"
          />
        </div>
        <div class="card-body">
          <div v-if="post.tags?.length" class="card-tags">
            <span v-for="tag in post.tags.slice(0, 3)" :key="tag" class="tag">{{ tag }}</span>
          </div>
          <h3 class="card-title">{{ post.title }}</h3>
          <span class="card-date">{{ formatTimestamp(post.date) }}</span>
        </div>
      </a>
    </div>
  </section>
</template>

<script setup>
import { formatTimestamp } from "@/utils/helper";
import { selectHomeFeed } from "@/utils/homeSections.mjs";
import { getPostPublicPath } from "@/utils/postUrl.mjs";

const { theme } = useData();
const props = defineProps({
  // 当前页数（只有首页第一页展示）
  page: {
    type: Number,
    default: 1,
  },
});

const featuredConfig = computed(() => theme.value?.home?.featured ?? {});
const title = computed(() => featuredConfig.value.title || "最新文章");
const moreLink = computed(() => featuredConfig.value.moreLink || "/pages/archives");

const featuredPosts = computed(() => {
  if (featuredConfig.value.enable === false) return [];
  if (props.page !== 1) return [];
  return selectHomeFeed(theme.value?.postData, theme.value).featured;
});
</script>

<style lang="scss" scoped>
.home-featured {
  width: 100%;
  margin-top: 2.6rem;

  .featured-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: var(--main-font-second-color);

    .featured-title {
      font-weight: 600;
      color: var(--main-font-color);
    }

    .featured-more {
      color: var(--main-color);
      font-weight: 600;

      &:hover {
        color: var(--main-color-hover);
      }
    }
  }

  .featured-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.25rem;
  }

  .featured-card {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 0;
    cursor: pointer;

    .card-cover {
      height: 150px;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: var(--main-card-second-background);
      border-bottom: 1px solid var(--main-card-border);
      overflow: hidden;

      img {
        max-width: 100%;
        max-height: 150px;
        display: block;
        transition: transform 0.4s;
      }
    }

    &:hover {
      .card-cover img {
        transform: scale(1.03);
      }
    }

    .card-body {
      display: flex;
      flex-direction: column;
      flex: 1;
      padding: 1rem 1.1rem 1.15rem;
    }

    .card-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 0.6rem;

      .tag {
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        color: var(--main-font-second-color);
        background-color: var(--main-card-second-background);
      }
    }

    .card-title {
      font-family: var(--main-serif-family);
      font-size: 1rem;
      font-weight: 600;
      line-height: 1.5;
      color: var(--main-font-color);
      margin: 0 0 0.7rem;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .card-date {
      margin-top: auto;
      font-size: 0.78rem;
      opacity: 0.6;
      color: var(--main-font-second-color);
    }
  }

  @media (max-width: 900px) {
    .featured-grid {
      grid-template-columns: 1fr;
    }

    .featured-card {
      flex-direction: row;

      .card-cover {
        width: 132px;
        flex: 0 0 132px;
        height: auto;
        min-height: 96px;
        border-bottom: none;
        border-right: 1px solid var(--main-card-border);
      }
    }
  }
}
</style>
