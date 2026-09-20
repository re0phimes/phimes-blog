/**
 * 页面浏览计数 - 使用 Cloudflare Workers KV
 */
import { ref, onMounted } from "vue";

// 注意：不要用 *.workers.dev —— 那个域名在国内被 DNS 污染，
// 解析到的是随机的社交网站 IP，浏览器请求会一直挂着不返回，
// 结果就是文章页的「热度」数字永远停在 0。
// 这里指向绑定了同一个 Worker 的自有域名（见 Cloudflare 的 Worker Route）。
const API_BASE = "https://views.phimes.top";

export function usePageViews(path) {
  const count = ref(0);
  const loading = ref(true);

  onMounted(async () => {
    if (!path) {
      loading.value = false;
      return;
    }

    try {
      // POST 增加计数并获取最新值
      const res = await fetch(`${API_BASE}?path=${encodeURIComponent(path)}`, {
        method: "POST",
      });
      const data = await res.json();
      count.value = data.count || 0;
    } catch (e) {
      console.error("Failed to update page views:", e);
    } finally {
      loading.value = false;
    }
  });

  return { count, loading };
}
