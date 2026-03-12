<!-- AI 摘要（假） -->
<template>
  <div v-if="frontmatter.articleGPT" class="article-gpt s-card">
    <div class="title">
      <span class="name" @click="showOther">
        <i class="iconfont icon-robot"></i>
        文章摘要
        <i class="iconfont icon-up"></i>
      </span>
      <span :class="['logo', { loading }]" @click="showOther"> FakeGPT </span>
    </div>
    <div class="content s-card">
      <span class="text">{{ abstractData === "" ? "加载中..." : abstractData }}</span>
      <span v-if="loading" class="point">|</span>
    </div>
    <div class="meta">
      <span class="tip">此内容根据文章生成，并经过人工审核，仅用于文章内容的解释与总结</span>
      <a
        href="https://eqnxweimkr5.feishu.cn/share/base/form/shrcnCXCPmxCKKJYI3RKUfefJre"
        class="report"
        target="_blank"
      >
        投诉
      </a>
    </div>
  </div>
</template>

<script setup>
const { frontmatter } = useData();

// 摘要数据
const loading = ref(false);
const abstractData = ref(frontmatter.value.articleGPT || "");
const showType = ref(false);

const fakeGptIntro =
  "我是無名开发的摘要生成助理 FakeGPT，如你所见，这是一个假的 GPT，所有文本皆源于本地书写的内容。我在这里只负责显示，并仿照 GPT 的形式输出，如果你像我一样囊中羞涩，你也可以像我这样做，当然，你也可以使用 Tianli 开发的 TianliGPT 来更简单地实现真正的 AI 摘要。";

const syncAbstract = () => {
  abstractData.value = showType.value ? fakeGptIntro : (frontmatter.value.articleGPT || "");
  loading.value = false;
};

// 输出摘要介绍
const showOther = () => {
  showType.value = !showType.value;
  syncAbstract();
};

watch(
  () => frontmatter.value.articleGPT,
  (value) => {
    showType.value = false;
    abstractData.value = value || "";
    loading.value = false;
  },
  { immediate: true },
);
</script>

<style lang="scss" scoped>
.article-gpt {
  margin-top: 1.2rem;
  background-color: var(--main-card-second-background);
  user-select: none;
  cursor: auto;
  .title {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.8rem;
    padding: 0 8px;
    .name {
      display: flex;
      align-items: center;
      color: var(--main-color);
      font-weight: bold;
      cursor: pointer;
      .icon-robot {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: normal;
        width: 26px;
        height: 26px;
        color: var(--main-card-background);
        background-color: var(--main-color);
        border-radius: 50%;
        margin-right: 8px;
      }
      .icon-up {
        font-weight: normal;
        font-size: 12px;
        margin-left: 6px;
        opacity: 0.6;
        color: var(--main-color);
        transform: rotate(90deg);
      }
    }
    .logo {
      padding: 4px 10px;
      font-size: 12px;
      color: var(--main-card-background);
      background-color: var(--main-color);
      border-radius: 25px;
      font-weight: bold;
      cursor: pointer;
      &.loading {
        animation: loading 1s infinite;
        cursor: not-allowed;
      }
    }
  }
  .content {
    cursor: auto;
    .point {
      color: var(--main-color);
      font-weight: bold;
      margin-left: 4px;
      animation: loading 0.8s infinite;
    }
  }
  .meta {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    margin-top: 1rem;
    padding: 0 8px;
    font-size: 12px;

    .tip {
      opacity: 0.6;
    }
    .report {
      white-space: nowrap;
      margin-left: 12px;
      opacity: 0.8;
    }
  }
}
</style>
