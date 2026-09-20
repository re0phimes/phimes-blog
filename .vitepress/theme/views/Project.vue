<template>
  <div class="project">
    <Banner type="page" title="我的项目" desc="做有挑战的事，走有壁垒的路" footer="Phimes">
      <template #footer-slot>
        <a v-if="githubLink" class="to-github" :href="githubLink" target="_blank" rel="noopener">
          <i class="iconfont icon-github"></i>
          <span>前往 Github</span>
        </a>
      </template>
    </Banner>

    <div class="project-list">
      <a
        v-for="(item, index) in projects"
        :key="index"
        class="project-card s-card hover"
        :href="item.link"
        target="_blank"
        rel="noopener"
      >
        <div class="card-head">
          <i :class="`iconfont icon-${item.icon}`" :style="{ color: item.color }" />
          <span class="card-name">{{ item.name }}</span>
        </div>
        <p class="card-desc">{{ item.desc }}</p>
        <div class="card-foot">
          <span class="card-tag">{{ item.tag }}</span>
          <i class="iconfont icon-right-round arrow" />
        </div>
      </a>
    </div>
  </div>
</template>

<script setup>
import Banner from "@/components/Banner.vue";

const { theme } = useData();

const projects = [
  {
    name: "Phimes",
    desc: "主站，几个站点和项目的入口。",
    link: "https://www.phimes.top/",
    tag: "入口",
    icon: "home",
    color: "#cc785c",
  },
  {
    name: "技术博客",
    desc: "你正在看的这个站。大模型与深度学习的学习记录，从注意力机制到推理优化。",
    link: "https://blog.phimes.top/",
    tag: "VitePress",
    icon: "article",
    color: "#5db8a6",
  },
  {
    name: "AI FAQ",
    desc: "AI 相关的问答知识库，把常见问题整理成可检索的条目。",
    link: "https://aifaq.phimes.top/",
    tag: "知识库",
    icon: "chat",
    color: "#e8a55a",
  },
  {
    name: "Demo",
    desc: "一些想法和效果的演示场，做到能跑起来为止。",
    link: "https://demo.phimes.top/",
    tag: "实验场",
    icon: "code",
    color: "#6a9bcc",
  },
];

const githubLink = computed(() => {
  const social = Array.isArray(theme.value?.footer?.social) ? theme.value.footer.social : [];
  return social.find((item) => item?.icon === "github")?.link || "";
});
</script>

<style lang="scss" scoped>
.project {
  .banner-page {
    .to-github {
      height: 40px;
      padding: 0 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50px;
      color: #fff;
      background-color: var(--main-dark-opacity);
      backdrop-filter: blur(20px);
      transition:
        color 0.3s,
        background-color 0.3s;
      .iconfont {
        margin-right: 8px;
        transition: color 0.3s;
      }
      &:hover {
        color: var(--main-card-background);
        background-color: var(--main-color);
        .iconfont {
          color: var(--main-card-background) !important;
        }
      }
    }
  }

  .project-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1.25rem;
    margin-top: 1.5rem;

    .project-card {
      display: flex;
      flex-direction: column;
      padding: 1.2rem 1.4rem;
      cursor: pointer;

      .card-head {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.6rem;

        .iconfont {
          font-size: 1.3rem;
        }

        .card-name {
          font-family: var(--main-serif-family);
          font-size: 1.2rem;
          font-weight: 600;
        }
      }

      .card-desc {
        margin: 0 0 1rem;
        font-size: 0.92rem;
        line-height: 1.75;
        color: var(--main-font-second-color);
      }

      .card-foot {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: auto;

        .card-tag {
          padding: 2px 10px;
          border-radius: 999px;
          font-size: 0.72rem;
          color: var(--main-font-second-color);
          background-color: var(--main-card-second-background);
        }

        .arrow {
          font-size: 0.8rem;
          opacity: 0.4;
          transition:
            opacity 0.3s,
            transform 0.3s;
        }
      }

      &:hover {
        .card-foot .arrow {
          opacity: 1;
          color: var(--main-color);
          transform: translateX(4px);
        }
      }
    }

    @media (max-width: 900px) {
      grid-template-columns: 1fr;
    }
  }
}
</style>
