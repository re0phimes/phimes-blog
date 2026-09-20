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
  "我是 Phimes 写的「假 GPT」——如你所见，这个名字就是字面意思：没有接任何模型，只是把文章摘要套了个对话式的外壳。上面那段摘要是我写文章时手写的，点标题可以在这段说明和摘要之间切换。想用真·AI 摘要的话，可以看看开源的本地摘要方案，或者直接调一家模型 API。";

const syncAbstract = () => {
  abstractData.value = showType.value ? fakeGptIntro : frontmatter.value.articleGPT || "";
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
