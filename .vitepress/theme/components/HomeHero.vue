<!-- 首页头条文章（Hero） -->
<template>
  <section v-if="hero" class="home-hero">
    <div class="hero-head">
      <span class="hero-label">{{ label }}</span>
      <span class="hero-meta">{{ formattedDate }}</span>
      <template v-if="heroTags.length">
        <span class="hero-meta dot">·</span>
        <span class="hero-meta">{{ heroTags.join(" / ") }}</span>
      </template>
    </div>
    <h1 class="hero-title">
      <a :href="heroPath">{{ hero.title || "未命名文章" }}</a>
    </h1>
    <div class="hero-foot">
      <p v-if="excerpt" class="hero-excerpt">{{ excerpt }}</p>
      <a class="hero-cta" :href="heroPath">阅读全文 →</a>
    </div>
  </section>
</template>

<script setup>
import { selectHomeHero } from "@/utils/homeSections.mjs";
import { getPostPublicPath } from "@/utils/postUrl.mjs";

const { theme } = useData();

const heroConfig = computed(() => theme.value?.home?.hero ?? {});
const hero = computed(() => selectHomeHero(theme.value?.postData, heroConfig.value));

const label = computed(() => heroConfig.value.label || "最新");
const heroTags = computed(() => (hero.value?.tags || []).slice(0, 4));
const heroPath = computed(() => (hero.value ? getPostPublicPath(hero.value) : "#"));

// 头条上用完整日期，比列表里的 M/D 更有仪式感
const formattedDate = computed(() => {
  if (!hero.value?.date) return "";
  const date = new Date(hero.value.date);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
});

// 摘要把 [[双链]] 和 $公式$ 还原成可读文字；优先用 themeConfig 里手写的
const cleanText = (input = "") =>
  String(input)
    .replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, target, alias) => alias || target)
    .replace(/\$\$?([^$]+)\$\$?/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/~~([^~]*)~~/g, "$1")
    // 自动摘要开头常是「1 前言」「TL;DR」这类小标题，读起来不像摘要，去掉
    .replace(/^\s*\d+(\.\d+)*\s*/, "")
    .replace(/^\s*(前言|TL;DR|摘要|引言|写在前面)\s*[:：]?\s*/i, "")
    .replace(/[#*`>]/g, "")
    .replace(/\s+/g, " ")
    .trim();

const excerpt = computed(() => {
  const written = cleanText(heroConfig.value.excerpt);
  if (written) return written;
  return cleanText(hero.value?.description).slice(0, 150);
});
</script>

<style lang="scss" scoped>
.home-hero {
  width: 100%;
  // 一条重线把头条从问候语里切出来，是整页视觉层级的起点
  border-top: 2px solid var(--main-font-color);
  margin-top: 1.5rem;
  padding: 1.6rem 0 0.5rem;
  animation: fade-up 0.6s 0.1s backwards;

  .hero-head {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 0.8rem;
    letter-spacing: 0.14em;
    color: var(--main-font-second-color);

    .hero-label {
      color: var(--main-color);
      font-weight: 700;
    }

    .hero-meta {
      letter-spacing: 0.02em;
    }

    .dot {
      opacity: 0.5;
    }
  }

  .hero-title {
    margin: 1.1rem 0 1.3rem;
    font-family: var(--main-serif-family);
    font-weight: 600;
    // 首屏主角：明显大于正文字号，建立视觉层级
    font-size: clamp(2rem, 3.6vw, 3.4rem);
    line-height: 1.24;
    letter-spacing: -0.02em;
    color: var(--main-font-color);

    a {
      color: inherit;
      transition: color 0.3s;

      &:hover {
        color: var(--main-color);
      }
    }
  }

  .hero-foot {
    display: flex;
    align-items: flex-end;
    gap: 3rem;

    .hero-excerpt {
      max-width: 640px;
      margin: 0;
      font-size: 1.05rem;
      line-height: 1.9;
      letter-spacing: 0.25px;
      color: var(--main-font-second-color);
    }

    .hero-cta {
      margin-left: auto;
      white-space: nowrap;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--main-color);

      &:hover {
        color: var(--main-color-hover);
      }
    }
  }

  @media (max-width: 768px) {
    padding-top: 1rem;

    .hero-foot {
      flex-direction: column;
      align-items: flex-start;
      gap: 1.2rem;

      .hero-cta {
        margin-left: 0;
      }
    }
  }
}
</style>
