<!-- 首页 -->
<template>
  <div class="home">
    <Banner v-if="showHeader" :height="store.bannerType" />
    <div class="home-content">
      <div class="posts-content">
        <!-- 头条文章 -->
        <HomeHero v-if="shouldShowHero" />
        <!-- 头条下方的封面次条 -->
        <HomeHighlights v-if="shouldShowHero" :page="Number(page)" />
        <!-- 分类 / 标签页：沿用服务端筛选 -->
        <template v-if="isTaxonomyPage">
          <TypeBar :type="showTags ? 'tags' : 'categories'" />
          <PostList :listData="postData" />
          <Pagination
            :total="allListTotal"
            :page="Number(page)"
            :limit="postSize"
            :useParams="true"
            :routePath="
              showCategories
                ? `/pages/categories/${showCategories}`
                : showTags
                  ? `/pages/tags/${showTags}`
                  : ''
            "
          />
        </template>

        <!-- 首页：历史文章（客户端排序 / 筛选 / 分页） -->
        <HomeArchive v-else />
      </div>
      <!-- 侧边栏 -->
      <Aside />
    </div>
  </div>
</template>

<script setup>
import { mainStore } from "@/store";
import { selectHomeFeed } from "@/utils/homeSections.mjs";

const { theme } = useData();
const store = mainStore();
const props = defineProps({
  // 显示首页头部
  showHeader: {
    type: Boolean,
    default: false,
  },
  // 当前页数
  page: {
    type: Number,
    default: 1,
  },
  // 显示分类
  showCategories: {
    type: [null, String],
    default: null,
  },
  // 显示标签
  showTags: {
    type: [null, String],
    default: null,
  },
});

// 每页文章数
const postSize = theme.value.postSize;

// 首页首屏数据（头条 + 封面次条 + 其余文章流，三者互不重复）
const homeFeedData = computed(() => selectHomeFeed(theme.value?.postData, theme.value));

// 首页文章流（按日期降序排序）
const homeFeedPostData = computed(() => {
  const { feed } = homeFeedData.value;
  if (!Array.isArray(feed)) return [];
  return feed.filter((post) => post && post.id !== undefined && post.id !== null);
});

// 首屏（头条 + 封面次条）显示条件：仅首页第一页，且不在分类/标签页
const shouldShowHero = computed(() => {
  if (props.showCategories || props.showTags) return false;
  // 默认仅首页第一页展示
  if (theme.value?.home?.featured?.onlyFirstPage === false) return true;
  return getCurrentPage() === 0;
});

// 分类 / 标签页沿用服务端筛选，首页走客户端的历史文章列表
const isTaxonomyPage = computed(() => Boolean(props.showCategories || props.showTags));

// 列表总数量
const allListTotal = computed(() => {
  const data = props.showCategories
    ? theme.value.categoriesData[props.showCategories]?.articles
    : props.showTags
      ? theme.value.tagsData[props.showTags]?.articles
      : homeFeedPostData.value;
  // 返回数量
  return data ? data.length : 0;
});

// 获得当前页数
const getCurrentPage = () => {
  if (props.showCategories || props.showTags) {
    if (typeof window === "undefined") return 0;
    const params = new URLSearchParams(window.location.search);
    const page = params.get("page");
    if (!page) return 0;
    const currentPage = Number(page);
    return currentPage ? currentPage - 1 : 0;
  }
  return props.page ? props.page - 1 : 0;
};

// 根据页数计算列表数据
const postData = computed(() => {
  const page = getCurrentPage();
  console.log("当前页数：", page);
  let data = null;
  // 分类数据
  if (props.showCategories) {
    data = theme.value.categoriesData[props.showCategories]?.articles;
  }
  // 标签数据
  else if (props.showTags) {
    data = theme.value.tagsData[props.showTags]?.articles;
  }
  // 文章数据
  else {
    data = homeFeedPostData.value;
  }
  // 返回列表
  return data ? data.slice(page * postSize, page * postSize + postSize) : [];
});

// 恢复滚动位置
const restoreScrollY = (val) => {
  if (typeof window === "undefined" || val) return false;
  const scrollY = store.lastScrollY;
  nextTick().then(() => {
    console.log("滚动位置：", scrollY);
    // 平滑滚动
    window.scrollTo({
      top: scrollY,
      behavior: "smooth",
    });
    // 清除滚动位置
    store.lastScrollY = 0;
  });
};

// 监听加载结束
watch(
  () => store.loadingStatus,
  (val) => restoreScrollY(val),
);
</script>

<style lang="scss" scoped>
.home {
  // 侧边栏宽度与主内容列间距。
  // 抽成变量是因为 HomeArchive 的翻页分区需要用它来和页脚对齐
  // （页脚是相对整个视口居中的，而主内容列被侧边栏挤窄了）。
  --home-aside-width: 300px;
  --home-content-gap: 1rem;

  .home-content {
    width: 100%;
    display: flex;
    flex-direction: row;
    gap: var(--home-content-gap);
    .posts-content {
      flex: 1;
      min-width: 0;
      transition: width 0.3s;
    }
    .main-aside {
      flex: 0 0 var(--home-aside-width);
      width: var(--home-aside-width);
      padding-left: 0;
    }
    @media (max-width: 1200px) {
      .posts-content {
        width: 100%;
      }
      .main-aside {
        display: none;
      }
    }
  }
}
</style>
