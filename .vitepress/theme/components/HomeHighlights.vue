<!-- 首页 Highlights：Most Popular -->
<template>
  <div v-if="enabled" class="home-highlights">
    <!-- Most Popular 横向卡片轮播 -->
    <section v-if="popularPostsWithCover.length" class="popular-carousel" aria-labelledby="home-highlights-popular">
      <div class="carousel-header">
        <span id="home-highlights-popular">Get started with our <strong>best stories</strong></span>
        <div class="carousel-nav">
          <button class="nav-btn" @click="scrollCarousel(-1)" aria-label="上一页">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
          </button>
          <button class="nav-btn" @click="scrollCarousel(1)" aria-label="下一页">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
        </div>
      </div>
      <div ref="carouselRef" class="carousel-track">
        <a
          v-for="post in popularPostsWithCover"
          :key="post.id ?? post.permalink ?? post.regularPath"
          :href="getPostPublicPath(post)"
          class="carousel-card"
        >
          <div class="card-cover">
            <img
              :src="post.cover"
              :alt="post.title"
              loading="lazy"
              @error="(e) => e.target.style.display = 'none'"
            />
          </div>
          <div class="card-tags" v-if="post.tags?.length">
            <span v-for="tag in post.tags.slice(0, 3)" :key="tag" class="tag">{{ tag }}</span>
          </div>
          <h3 class="card-title">{{ post.title }}</h3>
          <p class="card-desc" v-if="post.description">{{ post.description }}</p>
        </a>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from "vue";
import { getPostPublicPath } from "@/utils/postUrl.mjs";

const { theme } = useData();

const enabled = computed(() => Boolean(theme.value?.home?.highlights?.enable));

const getPostKey = (post) =>
  post?.id ?? post?.permalink ?? post?.regularPath ?? post?.legacyPath ?? post?.title;

const popularPostsWithCover = computed(() => {
  const posts = Array.isArray(theme.value?.postData) ? theme.value.postData : [];
  const mostPopularConfig = theme.value?.home?.highlights?.mostPopular ?? {};
  const rawLimit = Number(mostPopularConfig.limit);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 6;
  const curatedPaths = Array.isArray(mostPopularConfig.curated) ? mostPopularConfig.curated : [];

  const byPath = new Map();
  posts.forEach((post) => {
    if (!post) return;
    if (typeof post.regularPath === "string") byPath.set(post.regularPath, post);
    if (typeof post.legacyPath === "string") byPath.set(post.legacyPath, post);
    if (typeof post.permalink === "string") byPath.set(post.permalink, post);
  });

  const selected = [];
  const selectedKeys = new Set();
  const tryAdd = (post) => {
    if (!post?.cover) return false;
    const key = String(getPostKey(post));
    if (!key || selectedKeys.has(key)) return false;
    selected.push(post);
    selectedKeys.add(key);
    return true;
  };

  for (const path of curatedPaths) {
    if (selected.length >= limit) break;
    if (typeof path !== "string" || !path) continue;
    tryAdd(byPath.get(path));
  }

  const rankedPosts = posts
    .filter((post) => post?.cover && !selectedKeys.has(String(getPostKey(post))))
    .sort((a, b) => {
      const aRank = Number(a?.popularRank) || 0;
      const bRank = Number(b?.popularRank) || 0;
      if (aRank !== bRank) return bRank - aRank;

      const aDate = Number(a?.date) || 0;
      const bDate = Number(b?.date) || 0;
      if (aDate !== bDate) return bDate - aDate;

      return String(getPostKey(a)).localeCompare(String(getPostKey(b)));
    });

  for (const post of rankedPosts) {
    if (selected.length >= limit) break;
    tryAdd(post);
  }

  return selected;
});

// 轮播滚动
const carouselRef = ref(null);
const scrollCarousel = (dir) => {
  if (!carouselRef.value) return;
  const el = carouselRef.value;
  const cardWidth = 296;

  // 检测是否到头
  const atStart = el.scrollLeft <= 0;
  const atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 5;

  if ((dir < 0 && atStart) || (dir > 0 && atEnd)) {
    // 震动提示
    el.classList.add("shake");
    setTimeout(() => el.classList.remove("shake"), 300);
    return;
  }

  el.scrollBy({ left: dir * cardWidth, behavior: "smooth" });
};
</script>

<style lang="scss" scoped>
.home-highlights {
  margin-bottom: 1rem;
  animation: fade-up 0.6s 0.2s backwards;
}

.popular-carousel {
  margin-bottom: 1.5rem;
}

.carousel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  font-size: 1rem;
  opacity: 0.9;

  strong {
    font-weight: 700;
  }
}

.carousel-nav {
  display: flex;
  gap: 8px;

  .nav-btn {
    width: 32px;
    height: 32px;
    border: 1px solid var(--main-card-border);
    border-radius: 50%;
    background: var(--main-card-background);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    color: inherit;

    svg {
      opacity: 0.7;
    }

    &:hover {
      border-color: var(--main-color);
      svg {
        stroke: var(--main-color);
      }
    }
  }
}

.carousel-track {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scrollbar-width: none;
  -ms-overflow-style: none;
  &::-webkit-scrollbar {
    display: none;
  }

  &.shake {
    animation: shake 0.3s ease-in-out;
  }
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-8px); }
  75% { transform: translateX(8px); }
}

.carousel-card {
  flex: 0 0 calc((100% - 4rem) / 5);
  min-width: 240px;
  scroll-snap-align: start;
  display: flex;
  flex-direction: column;
  transition: transform 0.2s;

  &:hover {
    transform: translateY(-4px);
    .card-title {
      color: var(--main-color);
    }
  }
}

.card-cover {
  width: 100%;
  aspect-ratio: 4/3;
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 12px;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
}

.card-tags {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;

  .tag {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--main-card-second-background);
    opacity: 0.8;
  }
}

.card-title {
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.4;
  margin-bottom: 6px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  transition: color 0.2s;
}

.card-desc {
  font-size: 13px;
  opacity: 0.6;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
