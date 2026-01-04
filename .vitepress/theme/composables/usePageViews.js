/**
 * 页面浏览计数 - 使用 Cloudflare Workers KV
 */
import { ref, onMounted } from 'vue';

const API_BASE = 'https://blog-page-views.re0phimes.workers.dev';

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
        method: 'POST',
      });
      const data = await res.json();
      count.value = data.count || 0;
    } catch (e) {
      console.error('Failed to update page views:', e);
    } finally {
      loading.value = false;
    }
  });

  return { count, loading };
}
