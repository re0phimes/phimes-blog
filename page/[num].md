---
title: 正在重定向
aside: false
padding: false
---

<!--
  首页分页已改成客户端分页（/?p=N），/page/N 只保留为旧链接的跳板。
-->

<script setup>
import { onMounted } from "vue";
import { useData, useRouter } from "vitepress";

const { params } = useData();
const router = useRouter();

onMounted(() => {
  const num = Number(params.value.num);
  router.go(Number.isFinite(num) && num > 1 ? `/?p=${num}` : "/");
});
</script>

<p style="text-align: center; opacity: 0.6">正在跳转到历史文章…</p>
