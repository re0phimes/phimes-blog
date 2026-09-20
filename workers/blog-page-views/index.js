/**
 * blog-page-views —— 文章阅读量 API
 *
 * 挂在 views.phimes.top/*（比 site-analytics 的 *.phimes.top/* 更具体，优先级更高）
 *
 * ⚠️ 域名很关键：**不要用 *.workers.dev**。
 * 那个域名在国内被 DNS 污染（四个解析器返回四个不同的随机社交网站 IP），
 * 浏览器发出的请求根本到不了 Worker，表现就是文章页的「热度」数字恒为 0。
 * 必须绑自有域名，见 workers/README.md。
 *
 * 存储：Workers KV，namespace 绑定名 `PAGE_VIEWS`
 *   key   = 页面路径（如 /posts/2026/01/16/xxx）
 *   value = 计数字符串
 *
 * 接口：
 *   GET  /all?          → { "<path>": count, ... }   构建期 fetch-popular 用
 *   GET  /?path=<路径>  → { path, count }
 *   POST /?path=<路径>  → { path, count }            加一并返回最新值
 *
 * 注意：POST 是「无条件加一」，没有去重。同一个访客刷新一次就加一次，
 * 所以这个数字是 PV 不是 UV。真人/爬虫的区分在 site-analytics 那边做。
 */

export default {
  async fetch(request, env) {
    // CORS headers
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);
    const path = url.searchParams.get("path");

    // GET: 获取计数
    if (request.method === "GET") {
      // 获取所有计数（用于构建）
      if (url.pathname === "/all") {
        const list = await env.PAGE_VIEWS.list();
        const data = {};
        for (const key of list.keys) {
          data[key.name] = parseInt(await env.PAGE_VIEWS.get(key.name)) || 0;
        }
        return new Response(JSON.stringify(data), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      // 获取单个页面计数
      if (path) {
        const count = parseInt(await env.PAGE_VIEWS.get(path)) || 0;
        return new Response(JSON.stringify({ path, count }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
    }

    // POST: 增加计数
    if (request.method === "POST" && path) {
      const current = parseInt(await env.PAGE_VIEWS.get(path)) || 0;
      const newCount = current + 1;
      await env.PAGE_VIEWS.put(path, newCount.toString());
      return new Response(JSON.stringify({ path, count: newCount }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response("Bad Request", { status: 400, headers: corsHeaders });
  },
};
