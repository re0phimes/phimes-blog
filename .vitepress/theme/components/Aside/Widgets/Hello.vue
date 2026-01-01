<!-- 侧边栏 - 欢迎 -->
<template>
  <div class="hello s-card" @mouseleave="resetHello">
    <span class="tip" @click="changeHello">{{ helloText }}</span>
    <div class="info">
      <div class="name">
        <span class="author">{{ theme.siteMeta.author.name }}</span>
        <span class="desc">{{ theme.siteMeta.description }}</span>
      </div>
      <div class="link">
        <a href="https://github.com/imsyy/" target="_blank" class="social-link">
          <i class="iconfont icon-github"></i>
        </a>
        <a href="mailto:one@imsyy.top" target="_blank" class="social-link">
          <i class="iconfont icon-email"></i>
        </a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { getGreetings } from "@/utils/helper";

const { site, theme } = useData();

// 问候数据
const helloClick = ref(0);
const helloTimeOut = ref(null);
const helloText = ref(getGreetings());

// 恢复问候语
const resetHello = () => {
  helloClick.value = 0;
  if (isHasUser()) return false;
  helloText.value = getGreetings();
};

// 更改问候语
const changeHello = () => {
  clearTimeout(helloTimeOut.value);
  helloClick.value++;
  if (helloClick.value === 1) {
    helloText.value = "点这里干什么？";
  } else if (helloClick.value === 2) {
    helloText.value = "怎么还点？";
  } else if (helloClick.value === 3) {
    helloText.value = "那你点吧！";
  } else if (helloClick.value === 100) {
    helloText.value = "怎么还在点？？？";
  } else {
    helloText.value = `x ${helloClick.value - 3}`;
  }
  // 恢复默认
  helloTimeOut.value = setTimeout(() => {
    resetHello();
  }, 3000);
};

// 是否具有用户
const isHasUser = () => {
  // 检查本地存储
  const userData = localStorage.getItem("ArtalkUser");
  if (!userData) return false;
  // 获取用户数据
  const { nick } = JSON.parse(userData);
  const hello = ["很高兴见到你", "好久不见", "欢迎回来"];
  // 随机问候语
  helloText.value = hello[Math.floor(Math.random() * hello.length)] + "，" + nick;
  return true;
};

onMounted(() => {
  isHasUser();
});

onBeforeUnmount(() => {
  clearTimeout(helloTimeOut.value);
});
</script>

<style lang="scss" scoped>
.hello {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 12px;
  background-color: var(--main-color);
  color: var(--main-card-background);
  border: none;
  .tip {
    display: inline-block;
    min-width: 120px;
    text-align: center;
    padding: 2px 8px;
    border-radius: 25px;
    font-size: 14px;
    font-weight: bold;
    background-color: var(--main-color-opacity);
    margin-bottom: 4px;
    transition: color 0.3s, transform 0.3s, background-color 0.3s;
    &:hover {
      transform: scale(1.1);
      color: var(--main-font-color);
      background-color: var(--main-card-background);
    }
    &:active {
      transform: scale(1);
    }
  }
  .info {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    .name {
      display: flex;
      flex-direction: column;
      .author {
        font-weight: bold;
        font-size: 20px;
        line-height: 1.2;
      }
      .desc {
        font-size: 12px;
        opacity: 0.6;
      }
    }
    .link {
      display: flex;
      flex-direction: row;
      align-items: center;
      margin-left: 8px;
      .social-link {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        margin-left: 4px;
        background-color: var(--main-color-opacity);
        border-radius: 50%;
        .iconfont {
          font-size: 20px;
          color: var(--main-card-background);
        }
        &:first-child {
          margin-left: 0;
        }
        &:hover {
          transform: scale(1.1);
          background-color: var(--main-card-background);
          .iconfont {
            color: var(--main-font-color);
          }
        }
      }
    }
  }
}
</style>
