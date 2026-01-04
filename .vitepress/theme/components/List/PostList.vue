<!-- 文章列表 -->
<template>
  <div class="post-lists">
    <a
      v-for="(item, index) in listData"
      :key="index"
      :href="item.regularPath"
      class="post-item s-card hover"
      :style="{ animationDelay: `${0.4 + index / 10}s` }"
    >
      <span class="rank">{{ index + 1 }}</span>
      <span class="title">{{ item.title }}</span>
      <span class="date">{{ formatTimestamp(item?.date) }}</span>
    </a>
  </div>
</template>

<script setup>
import { formatTimestamp } from "@/utils/helper";

defineProps({
  listData: {
    type: [Array, String],
    default: () => [],
  },
});
</script>

<style lang="scss" scoped>
.post-lists {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.post-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 12px 16px;
  animation: fade-up 0.6s 0.4s backwards;
  cursor: pointer;
  transition: border-color 0.3s, box-shadow 0.3s, background-color 0.3s;

  .rank {
    width: 28px;
    flex: 0 0 28px;
    font-weight: bold;
    opacity: 0.75;
    color: var(--main-color);
  }

  .title {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding-right: 12px;
    font-weight: 500;
  }

  .date {
    font-size: 13px;
    opacity: 0.6;
    white-space: nowrap;
  }

  &:hover {
    border-color: var(--main-color);
    box-shadow: 0 8px 16px -4px var(--main-color-bg);
    background-color: var(--main-color-bg);
    .title {
      color: var(--main-color);
    }
  }

  &:active {
    transform: scale(0.99);
  }
}
</style>
