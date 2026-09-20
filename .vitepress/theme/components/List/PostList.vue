<!-- 文章列表 -->
<template>
  <div class="post-lists">
    <a
      v-for="(item, index) in listData"
      :key="index"
      :href="item.permalink || item.regularPath"
      class="post-item s-card hover"
      :style="{ animationDelay: `${0.4 + index / 10}s` }"
    >
      <span class="rank">{{ startIndex + index + 1 }}</span>
      <span class="title">{{ item.title }}</span>
      <span class="date-area">
        <span v-if="showPopular && Number.isFinite(Number(item.popularRank))" class="popular-badge">
          {{ item.popularRank }} 次浏览
        </span>
        <span class="date">{{ formatTimestamp(item?.date) }}</span>
        <span v-if="item.version > 1" class="version-badge">v{{ item.version }}</span>
        <span v-if="item.lastUpdated && item.lastUpdated !== item.date" class="updated-badge"
          >更新于 {{ formatShortDate(item.lastUpdated) }}</span
        >
      </span>
    </a>
  </div>
</template>

<script setup>
import { formatTimestamp } from "@/utils/helper";

const formatShortDate = (ts) => {
  const d = new Date(ts);
  const now = new Date();
  const m = d.getMonth() + 1;
  const day = d.getDate();
  if (d.getFullYear() === now.getFullYear()) return `${m}/${day}`;
  return `${d.getFullYear()}/${m}/${day}`;
};

defineProps({
  listData: {
    type: [Array, String],
    default: () => [],
  },
  // 列表起始序号（客户端分页时用来让序号连续）
  startIndex: {
    type: Number,
    default: 0,
  },
  // 是否显示浏览量（按热度排序时用来让排序结果可读）
  showPopular: {
    type: Boolean,
    default: false,
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
  transition:
    border-color 0.3s,
    box-shadow 0.3s,
    background-color 0.3s;

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

  .date-area {
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .date {
    font-size: 13px;
    opacity: 0.6;
    white-space: nowrap;
  }

  .version-badge {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--main-color);
    color: #fff;
    font-weight: 600;
    line-height: 1.4;
  }

  .popular-badge {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--main-color-bg);
    color: var(--main-color);
    font-weight: 600;
    line-height: 1.4;
  }

  .updated-badge {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--main-color-bg);
    color: var(--main-color);
    font-weight: 500;
    line-height: 1.4;
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

  // 窄屏：日期单独占一行，标题才能有两行完整宽度
  // （否则「2025/9/28 更新于 2/27」会把标题挤到只剩十来个字）
  @media (max-width: 768px) {
    align-items: flex-start;
    flex-wrap: wrap;
    row-gap: 4px;

    .title {
      flex: 1 1 auto;
      width: calc(100% - 28px);
      white-space: normal;
      padding-right: 0;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      line-height: 1.5;
    }

    .date-area {
      width: 100%;
      padding-left: 28px;
      gap: 6px;
    }

    .date,
    .popular-badge,
    .version-badge,
    .updated-badge {
      font-size: 11.5px;
    }
  }
}
</style>
