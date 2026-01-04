<template>
  <Teleport to="body">
    <!-- 站点背景 -->
    <div :class="['background', themeValue, { paused: !bgAnimationEnabled }]">
      <div class="bottom-mask"></div>
    </div>
  </Teleport>
</template>

<script setup>
import { storeToRefs } from "pinia";
import { withBase } from "vitepress";
import { mainStore } from "@/store";

const store = mainStore();
const { themeValue, bgAnimationEnabled } = storeToRefs(store);
</script>

<style lang="scss" scoped>
.background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: -2;
  background-color: var(--main-site-background);
  background-size: 60px 60px;
  animation: gridMove 8s linear infinite;

  &.paused {
    animation-play-state: paused;
  }

  &.dark {
    background-image:
      linear-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.08) 1px, transparent 1px);
  }
  &.light {
    background-image:
      linear-gradient(rgba(0, 0, 0, 0.06) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 0, 0, 0.06) 1px, transparent 1px);
  }

  .bottom-mask {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 50%;
    background: linear-gradient(to bottom, transparent, var(--main-site-background));
    pointer-events: none;
  }
}

@keyframes gridMove {
  from { background-position: 0 0, 0 0; }
  to { background-position: 60px 60px, 60px 60px; }
}
</style>
