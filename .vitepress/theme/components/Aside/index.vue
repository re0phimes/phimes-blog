<template>
  <aside class="main-aside">
    <div class="sticky">
      <Toc v-if="theme.aside.toc.enable && showToc" class="weidgets" />
      <!--
        标签组件：
        - 文章页（context="post"）：只显示「相关标签」，且没有相关标签时整个隐藏
        - 其他页面：跟随 aside.tags.enable（首页已关掉热门标签）
      -->
      <Tags v-if="showTags" class="weidgets" :context="context" :relatedOnly="isPostAside" />
      <SiteData v-if="theme.aside.siteData.enable" class="weidgets" />
    </div>
  </aside>
</template>

<script setup>
const { theme } = useData();
const props = defineProps({
  // 显示目录
  showToc: {
    type: Boolean,
    default: false,
  },
  // 标签组件上下文：home = 全局热门；post = 与当前文章相关
  context: {
    type: String,
    default: "home",
  },
});

// 文章页侧边栏：只保留「相关标签」
const isPostAside = computed(() => props.context === "post");

// 文章页不受 aside.tags.enable 影响（那是首页热门标签的开关），
// 其余页面跟随配置
const showTags = computed(() => isPostAside.value || theme.value.aside.tags.enable);
</script>

<style lang="scss" scoped>
.main-aside {
  padding-left: 1rem;
  display: flex;
  flex-direction: column;
  animation: fade-up 0.6s 0.3s backwards;
  .weidgets {
    padding: 18px;
    margin-bottom: 1rem;
    :deep(.title) {
      margin-bottom: 12px;
      font-weight: bold;
      display: flex;
      align-items: center;
      opacity: 0.75;
      .iconfont {
        opacity: 0.6;
        margin-right: 6px;
      }
      .title-name {
        opacity: 0.8;
      }
    }
  }
  .sticky {
    position: sticky;
    top: calc(60px + 1rem);
    .weidgets {
      animation: fade-up 0.6s 0.4s backwards;
      &:last-child {
        margin-bottom: 0;
      }
    }
  }
}
</style>
