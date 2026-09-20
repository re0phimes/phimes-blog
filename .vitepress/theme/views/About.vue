<template>
  <div class="about">
    <h1 class="title">关于本站</h1>
    <div class="about-content" style="grid-template-columns: 3fr 2fr">
      <!-- 介绍 -->
      <div class="about-item hello">
        <span class="text1">你好，很高兴认识你 👋</span>
        <span class="text2 title2">我是 Phimes</span>
        <span class="text3">做有挑战的事，走有壁垒的路</span>
      </div>
      <!-- 关注 -->
      <div class="about-item pursuit">
        <span class="tip">关注</span>
        <span class="title2">在 AI 时代</span>
        <span class="title2">思考与分享</span>
      </div>
    </div>
    <div class="about-content" style="grid-template-columns: 2fr 3fr">
      <!-- 技术栈 -->
      <div class="about-item skills">
        <span class="tip">技术栈</span>
        <span class="title2">常用的工具</span>
        <div class="skills-list">
          <a
            v-for="(item, index) in skillsData"
            :key="index"
            :style="{ '--color': item.color }"
            :href="item.link"
            class="skills-item"
            target="_blank"
            rel="noopener"
          >
            <div class="skills-logo">
              <i :class="`iconfont icon-${item.icon}`"></i>
            </div>
            <span class="skills-name">{{ item.name }}</span>
          </a>
        </div>
      </div>
      <!-- 站点 -->
      <div class="about-item sites">
        <span class="tip">站点</span>
        <span class="title2">几个自己写的小东西</span>
        <div class="site-list">
          <a
            v-for="(site, index) in siteData"
            :key="index"
            :href="site.link"
            class="site-item"
            target="_blank"
            rel="noopener"
          >
            <span class="site-name">{{ site.name }}</span>
            <span class="site-desc">{{ site.desc }}</span>
            <i class="iconfont icon-right-round site-arrow" />
          </a>
        </div>
      </div>
    </div>
    <div class="about-content" style="grid-template-columns: 1fr 1fr">
      <!-- 座右铭 -->
      <div class="about-item motto">
        <span class="tip">座右铭</span>
        <span class="title1">做有挑战的事，</span>
        <span class="title2">走有壁垒的路。</span>
      </div>
      <!-- 本站数据 -->
      <div class="about-item static" style="--color: #0f1114">
        <div class="image-content">
          <span class="tip">数据</span>
          <span class="title2">这里写了多少</span>
          <div class="static-data">
            <div class="static-item">
              <span class="static-name">文章</span>
              <span class="static-num">{{ postCount }}</span>
            </div>
            <div class="static-item">
              <span class="static-name">标签</span>
              <span class="static-num">{{ tagCount }}</span>
            </div>
            <div class="static-item">
              <span class="static-name">建站于</span>
              <span class="static-num small">{{ since }}</span>
            </div>
            <div class="static-item">
              <span class="static-name">累计</span>
              <span class="static-num small">{{ runningDays }} 天</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- 心路历程 -->
    <div class="about-content" style="display: flex">
      <div class="about-item">
        <span class="tip">心路历程</span>
        <span class="title2">为什么建站？</span>
        <p class="text">
          创建这个站的时候，想要就是能够有一个自己能够<strong>积累知识</strong>、<strong>积累兴趣</strong>的地方。和他人分享，会让这些成为<strong>积累和沉淀</strong>。如果能够帮助到更多的人，帮助更多人解决问题，那一定是非常棒的事情。
        </p>
        <p class="text">
          这里大多都是<strong>大模型与深度学习</strong>相关的文章——从 Transformer 的注意力机制、KV
          Cache 的推理优化，到微调与部署的工程实践。写得比较慢，但尽量把每一步算清楚。
        </p>
        <p class="text">
          这些就是创造这个小站的本意，<strong>也是我分享生活的方式</strong>。有幸能和你相遇在这里，相信我们能共同留下一段美好记忆。
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
const { theme } = useData();

// 技术栈（图标使用主题自带 iconfont 中已有的名字）
const skillsData = [
  { name: "Python", color: "#3776AB", icon: "python", link: "https://www.python.org/" },
  { name: "PyTorch", color: "#EE4C2C", icon: "python", link: "https://pytorch.org/" },
  {
    name: "JavaScript",
    color: "#f1e05abd",
    icon: "javascript",
    link: "https://developer.mozilla.org/zh-CN/docs/Web/JavaScript",
  },
  { name: "Vue", color: "#41b883", icon: "vue", link: "https://cn.vuejs.org/" },
  { name: "Node.js", color: "#026E00", icon: "nodejs", link: "https://nodejs.org/" },
  { name: "Docker", color: "#2496f2", icon: "docker", link: "https://www.docker.com/" },
  { name: "Git", color: "#F05032", icon: "git", link: "https://git-scm.com/" },
  { name: "LLM", color: "#4AA181", icon: "chatgpt", link: "https://www.deepseek.com/" },
];

// 我自己的站点
const siteData = [
  { name: "主站", desc: "入口与导航", link: "https://www.phimes.top/" },
  { name: "技术博客", desc: "就是你现在看的这个", link: "https://blog.phimes.top/" },
  { name: "AI FAQ", desc: "AI 问答知识库", link: "https://aifaq.phimes.top/" },
  { name: "Demo", desc: "项目演示", link: "https://demo.phimes.top/" },
];

// 本站数据（全部来自构建期生成的真实数据）
const postCount = computed(() => theme.value?.postData?.length ?? 0);
const tagCount = computed(() => Object.keys(theme.value?.tagsData ?? {}).length);
const since = computed(() => theme.value?.since || "");
const runningDays = computed(() => {
  const start = new Date(theme.value?.since || "");
  if (Number.isNaN(start.getTime())) return "—";
  return Math.max(0, Math.floor((Date.now() - start.getTime()) / 86400000));
});
</script>

<style lang="scss" scoped>
.about {
  .title {
    font-size: 2.4rem;
    text-align: center;
    border: none;
  }
  .about-content {
    display: grid;
    grid-template-columns: auto auto;
    gap: 20px;
    margin-bottom: 20px;
    .about-item {
      position: relative;
      display: flex;
      flex-direction: column;
      width: 100%;
      padding: 1.2rem 2rem;
      border-radius: 12px;
      background-color: var(--main-card-background);
      border: 1px solid var(--main-card-border);
      box-shadow: 0 8px 12px -4px var(--main-border-shadow);
      overflow: hidden;
      .tip {
        font-size: 14px;
        opacity: 0.8;
        margin-bottom: 12px;
      }
      .title1 {
        font-size: 36px;
        font-weight: bold;
        opacity: 0.6;
      }
      .title2 {
        font-size: 36px;
        font-weight: bold;
        margin-right: 4rem;
      }
      .text {
        font-size: 18px;
        margin: 0.6rem 0;
      }
      &.hello {
        justify-content: center;
        padding: 2rem;
        color: #fff;
        background-image: linear-gradient(120deg, #5b27ff 0%, #00d4ff 100%);
        background-size: 200% 200%;
        animation: gradientFlow 6s ease infinite;
        .title2 {
          line-height: 2;
        }
      }
      &.pursuit {
        .title2 {
          line-height: 1.2;
          &:last-child {
            display: inline-block;
            background-size: 100% 100%;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-repeat: no-repeat;
            background-image: linear-gradient(45deg, #fa7671 50%, #f45f7f);
          }
        }
      }
      // 座右铭：用渐变代替原来失效的背景图
      &.motto {
        justify-content: center;
        min-height: 180px;
        background-image: linear-gradient(135deg, #0c0e20 0%, #2b3a55 100%);
        color: #fff;
        .tip {
          opacity: 0.6;
        }
        .title1 {
          opacity: 1;
        }
        .title2 {
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-image: linear-gradient(45deg, #7fd8ff 0%, #b18cff 100%);
        }
      }
      &.skills {
        .skills-list {
          margin-top: 12px;
          display: flex;
          flex-direction: row;
          flex-wrap: wrap;
          .skills-item {
            display: flex;
            align-items: center;
            margin-right: 10px;
            margin-top: 10px;
            padding: 8px 12px 8px 8px;
            border-radius: 40px;
            background-color: var(--main-site-background);
            border: 1px solid var(--main-card-border);
            box-shadow: 0 8px 12px -4px var(--main-border-shadow);
            transition: background-color 0.3s;
            cursor: pointer;
            .skills-logo {
              display: flex;
              align-items: center;
              justify-content: center;
              width: 32px;
              height: 32px;
              margin-right: 8px;
              border-radius: 50%;
              background-color: var(--color);
              .iconfont {
                color: #fff;
              }
            }
            .skills-name {
              font-weight: bold;
              transition: color 0.3s;
            }
            &:hover {
              background-color: var(--main-card-background);
            }
          }
        }
      }
      // 站点列表（替代原来失效的生涯配图）
      &.sites {
        .site-list {
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .site-item {
          display: flex;
          align-items: baseline;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 10px;
          background-color: var(--main-site-background);
          border: 1px solid var(--main-card-border);
          transition:
            border-color 0.3s,
            background-color 0.3s,
            transform 0.3s;
          .site-name {
            font-weight: bold;
          }
          .site-desc {
            font-size: 13px;
            opacity: 0.65;
          }
          .site-arrow {
            margin-left: auto;
            font-size: 12px;
            opacity: 0.4;
            transition: opacity 0.3s;
          }
          &:hover {
            border-color: var(--main-color);
            background-color: var(--main-color-bg);
            transform: translateX(4px);
            .site-arrow {
              opacity: 1;
              color: var(--main-color);
            }
          }
        }
      }
      &.static {
        .static-data {
          display: grid;
          gap: 12px;
          grid-template-columns: 1fr 1fr;
          margin: 20px 0;
          .static-item {
            display: flex;
            flex-direction: column;
            .static-name {
              font-size: 15px;
              opacity: 0.8;
            }
            .static-num {
              font-size: 34px;
              font-weight: bold;
              &.small {
                font-size: 20px;
              }
            }
          }
        }
      }
    }
    &:last-child {
      margin-bottom: 0;
    }
    @media (max-width: 768px) {
      display: flex;
      flex-direction: column;
    }
  }
}
</style>
