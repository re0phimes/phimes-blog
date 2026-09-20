import { defineConfig } from "vitepress";
import { createRssFile } from "./theme/utils/generateRSS.mjs";
import { createBlogIndex } from "./theme/utils/generateBlogIndex.mjs";
import { createLlmsTxt } from "./theme/utils/generateLlmsTxt.mjs";
import { withPwa } from "@vite-pwa/vitepress";
import {
  getAllPosts,
  getAllType,
  getAllCategories,
  getAllArchives,
} from "./theme/utils/getPostData.mjs";
import { jumpRedirect } from "./theme/utils/commonTools.mjs";
import { getThemeConfig } from "./init.mjs";
import markdownConfig from "./theme/utils/markdownConfig.mjs";
import AutoImport from "unplugin-auto-import/vite";
import Components from "unplugin-vue-components/vite";
import fs from "fs-extra";
import path from "path";
import {
  buildLegacyRedirectHtml,
  buildPageViewPathCandidates,
  buildPostRewriteRules,
  buildPostUrlData,
} from "./theme/utils/postUrl.mjs";
import { transformSitemapItems } from "./theme/utils/sitemap.mjs";

// 获取全局数据
const postData = await getAllPosts();

// 获取主题配置
const themeConfig = await getThemeConfig();
const algoliaHost = "https://X5EBEZB53I-dsn.algolia.net";

const resolvedHeadEntries = (themeConfig.inject?.header || []).filter((entry) => {
  if (!Array.isArray(entry) || entry.length < 2) return true;
  const attrs = entry[1] || {};
  const href = attrs.href;
  if (typeof href !== "string" || !href) return true;

  if (!themeConfig?.search?.enable && href === algoliaHost) return false;

  return true;
});

/**
 * Optional: merge Most Popular ranking cache into postData at build-time.
 *
 * Supported JSON shapes (examples):
 * - [\"/posts/a.html\", \"/posts/b.html\"]                      // order list
 * - [{\"regularPath\":\"/posts/a.html\",\"score\":123}, ...]   // score list
 * - {\"/posts/a.html\": 123, \"/posts/b.html\": 80}            // score map
 *
 * This is best-effort: missing file / parse errors will NOT fail the build.
 */
const mergeMostPopularExternalCache = async (posts, config) => {
  const cacheConfig = config?.home?.highlights?.mostPopular?.externalCache;
  if (!cacheConfig?.enable) return;

  const rawFile = cacheConfig.file || "data/popular.json";
  const filePath = path.resolve(__dirname, "..", rawFile);

  const exists = await fs.pathExists(filePath);
  if (!exists) {
    console.warn(`[popular-cache] file not found, fallback to curated: ${rawFile}`);
    return;
  }

  let raw;
  try {
    raw = await fs.readJSON(filePath);
  } catch (error) {
    console.warn(`[popular-cache] failed to read/parse ${rawFile}, fallback to curated`, error);
    return;
  }

  const scoreByPath = new Map();

  // 1) order list
  if (Array.isArray(raw) && raw.every((item) => typeof item === "string")) {
    raw.forEach((regularPath, index) => {
      // Higher score ranks higher
      scoreByPath.set(regularPath, raw.length - index);
    });
  }
  // 2) score list
  else if (Array.isArray(raw) && raw.every((item) => item && typeof item === "object")) {
    raw.forEach((item) => {
      const regularPath = item.regularPath || item.path;
      const score = item.score ?? item.rank ?? item.pv ?? item.views;
      const scoreNumber = Number(score);
      if (typeof regularPath !== "string" || !regularPath) return;
      if (!Number.isFinite(scoreNumber)) return;
      scoreByPath.set(regularPath, scoreNumber);
    });
  }
  // 3) score map
  else if (raw && typeof raw === "object") {
    Object.entries(raw).forEach(([regularPath, score]) => {
      const scoreNumber = Number(score);
      if (typeof regularPath !== "string" || !regularPath) return;
      if (!Number.isFinite(scoreNumber)) return;
      scoreByPath.set(regularPath, scoreNumber);
    });
  }

  if (scoreByPath.size === 0) {
    console.warn(`[popular-cache] empty/invalid cache: ${rawFile}`);
    return;
  }

  posts.forEach((post) => {
    if (!post) return;
    const score = buildPageViewPathCandidates(post).reduce((sum, currentPath) => {
      const currentScore = Number(scoreByPath.get(currentPath));
      return sum + (Number.isFinite(currentScore) ? currentScore : 0);
    }, 0);
    if (!Number.isFinite(score) || score <= 0) return;
    post.popularRank = score;
    post.popular = true;
  });
};

await mergeMostPopularExternalCache(postData, themeConfig);
const postRewriteMap = buildPostRewriteRules(postData);

// https://vitepress.dev/reference/site-config
export default withPwa(
  defineConfig({
    title: themeConfig.siteMeta.title,
    description: themeConfig.siteMeta.description,
    lang: themeConfig.siteMeta.lang,
    // 简洁的 URL
    cleanUrls: true,
    rewrites: (page) => postRewriteMap[page],
    // 最后更新时间戳
    lastUpdated: true,
    // 主题
    appearance: "dark",
    head: [
      ...resolvedHeadEntries,

      // 说明：这里原来挂着 Microsoft Clarity 的会话录制脚本（Project ID ty1lrwm16k）。
      // 它会记录访客的点击、滚动、输入并回放到微软服务器，属于隐私成本最高的一类统计，
      // 而且从国内加载很慢。已移除。
      // 想继续看流量，用 Cloudflare Web Analytics（免费、无 cookie、不采集个人数据）：
      //   Cloudflare 控制台 → Analytics & Logs → Web Analytics → 添加站点 → 复制 beacon
      //
      // 全站 PV/UV 目前仍由侧边栏的 busuanzi 提供（见 Aside/Widgets/SiteData.vue），
      // 文章阅读量由自建 Worker 提供（见 composables/usePageViews.js）。
    ],

    // sitemap
    sitemap: {
      hostname: themeConfig.siteMeta.site,
      transformItems: (items) => {
        const result = transformSitemapItems(items, postData, {
          site: themeConfig.siteMeta.site,
        });
        const transformedCount = result.filter(
          (item) => item.url.startsWith("posts/") && item.url.includes("/20"),
        ).length;
        console.log(`[Sitemap] Kept ${result.length} items after cleanup`);
        console.log(`[Sitemap] Normalized ${transformedCount} post URLs to SEO-friendly format`);
        return result;
      },
    },
    // 主题配置
    themeConfig: {
      ...themeConfig,
      // 必要数据
      postData,
      tagsData: getAllType(postData),
      categoriesData: getAllCategories(postData),
      archivesData: getAllArchives(postData),
    },
    // markdown
    markdown: {
      math: false,
      lineNumbers: false,
      toc: { level: [1, 2, 3] },
      image: {
        lazyLoading: true,
      },
      config: (md) =>
        markdownConfig(md, themeConfig, {
          posts: postData,
          tags: Object.keys(getAllType(postData)),
        }),
    },
    // 构建排除
    srcExclude: ["**/README.md", "**/TODO.md", "**/TEST_REPORT.md", "docs/plans/**", "plan/**"],
    // transformHead
    transformPageData: async (pageData) => {
      // 新增：URL 结构优化
      // 只处理 posts 目录下的文章
      let customUrl = null;
      if (pageData.relativePath.startsWith("posts/")) {
        const frontmatter = pageData.frontmatter;
        const urlData = buildPostUrlData({
          relativePath: pageData.relativePath,
          date: frontmatter.date,
          topic: frontmatter.topic,
          tags: frontmatter.tags,
        });

        if (urlData) {
          frontmatter.id = urlData.id;
          customUrl = urlData.permalink;
          frontmatter.permalink = customUrl;
        }
      }

      // canonical URL（使用自定义 URL 或默认 URL）
      const canonicalUrl = customUrl
        ? `${themeConfig.siteMeta.site}${customUrl}`
        : `${themeConfig.siteMeta.site}/${pageData.relativePath}`
            .replace(/index\.md$/, "")
            .replace(/\.md$/, "");

      pageData.frontmatter.head ??= [];
      pageData.frontmatter.head.push(["link", { rel: "canonical", href: canonicalUrl }]);

      // JSON-LD 结构化数据。
      // 搜索引擎靠它出富摘要，AI 爬虫（GPTBot / ClaudeBot / PerplexityBot 等）
      // 也用它来判断「这是什么页面、作者是谁、什么时候写的」，
      // 比从 HTML 里猜要准得多。
      const fm = pageData.frontmatter || {};
      const isPost = pageData.relativePath.startsWith("posts/");
      const site = themeConfig.siteMeta.site.replace(/\/$/, "");

      const publisher = {
        "@type": "Person",
        name: themeConfig.siteMeta.author?.name || themeConfig.siteMeta.title,
        url: site,
        ...(themeConfig.siteMeta.author?.link
          ? { sameAs: [themeConfig.siteMeta.author.link] }
          : {}),
      };

      const graph = [
        {
          "@type": "WebSite",
          "@id": `${site}/#website`,
          url: `${site}/`,
          name: themeConfig.siteMeta.title,
          description: themeConfig.siteMeta.description,
          inLanguage: "zh-CN",
          publisher,
        },
      ];

      if (isPost) {
        graph.push({
          "@type": "BlogPosting",
          "@id": `${canonicalUrl}#article`,
          mainEntityOfPage: canonicalUrl,
          headline: fm.title || pageData.title,
          description: fm.description || pageData.description || "",
          inLanguage: "zh-CN",
          ...(fm.date ? { datePublished: new Date(fm.date).toISOString() } : {}),
          ...(fm.lastUpdated ? { dateModified: new Date(fm.lastUpdated).toISOString() } : {}),
          ...(Array.isArray(fm.tags) && fm.tags.length ? { keywords: fm.tags.join(", ") } : {}),
          ...(Array.isArray(fm.categories) && fm.categories.length
            ? { articleSection: fm.categories }
            : {}),
          ...(fm.cover ? { image: fm.cover } : {}),
          author: publisher,
          publisher,
          isPartOf: { "@id": `${site}/#website` },
        });
      }

      pageData.frontmatter.head.push([
        "script",
        { type: "application/ld+json" },
        JSON.stringify({ "@context": "https://schema.org", "@graph": graph }),
      ]);
    },
    // transformHtml
    transformHtml: (html) => {
      return jumpRedirect(html, themeConfig);
    },
    // buildEnd
    buildEnd: async (config) => {
      await createRssFile(config, themeConfig);
      await createBlogIndex(config, themeConfig);
      await createLlmsTxt(config, themeConfig);

      console.log("[URL Optimization] Generating legacy redirect pages...");
      let redirectCount = 0;

      for (const post of postData) {
        if (!post?.legacyPath || !post?.permalink) continue;

        const legacyHtmlPath = path.join(config.outDir, post.legacyPath.replace(/^\//, ""));
        await fs.ensureDir(path.dirname(legacyHtmlPath));
        await fs.writeFile(legacyHtmlPath, buildLegacyRedirectHtml(post.permalink), "utf-8");
        redirectCount++;
      }

      console.log(`[URL Optimization] Generated ${redirectCount} legacy redirect pages`);
    },
    // vite
    vite: {
      plugins: [
        AutoImport({
          imports: ["vue", "vitepress"],
          dts: ".vitepress/auto-imports.d.ts",
        }),
        Components({
          dirs: [".vitepress/theme/components", ".vitepress/theme/views"],
          extensions: ["vue", "md"],
          include: [/\.vue$/, /\.vue\?vue/, /\.md$/],
          dts: ".vitepress/components.d.ts",
        }),
      ],
      resolve: {
        // 配置路径别名
        alias: {
          // eslint-disable-next-line no-undef
          "@": path.resolve(__dirname, "./theme"),
        },
      },
      css: {
        preprocessorOptions: {
          scss: {
            silenceDeprecations: ["legacy-js-api"],
          },
        },
      },
      // 服务器
      server: {
        port: 9877,
      },
      // 构建
      build: {
        minify: "terser",
        terserOptions: {
          compress: {
            pure_funcs: ["console.log"],
          },
        },
      },
    },
    // PWA
    pwa: {
      registerType: "autoUpdate",
      selfDestroying: true,
      workbox: {
        clientsClaim: true,
        skipWaiting: true,
        cleanupOutdatedCaches: true,
        // 资源缓存
        runtimeCaching: [
          {
            urlPattern: /(.*?)\.(woff2|woff|ttf|css)/,
            handler: "CacheFirst",
            options: {
              cacheName: "file-cache",
            },
          },
          {
            urlPattern: /(.*?)\.(ico|webp|png|jpe?g|svg|gif|bmp|psd|tiff|tga|eps)/,
            handler: "CacheFirst",
            options: {
              cacheName: "image-cache",
            },
          },
          {
            urlPattern: /^https:\/\/cdn2\.codesign\.qq\.com\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "iconfont-cache",
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 2,
              },
              cacheableResponse: {
                statuses: [0, 200],
              },
            },
          },
        ],
        // 缓存文件
        globPatterns: ["**/*.{js,css,html,ico,png,jpg,jpeg,gif,svg,woff2,ttf}"],
        // 排除路径
        navigateFallbackDenylist: [/^\/sitemap.xml$/, /^\/rss.xml$/, /^\/robots.txt$/],
      },
      manifest: {
        name: themeConfig.siteMeta.title,
        short_name: themeConfig.siteMeta.title,
        description: themeConfig.siteMeta.description,
        display: "standalone",
        start_url: "/",
        theme_color: "#fff",
        background_color: "#efefef",
        icons: [
          {
            src: "/images/logo/favicon-32x32.webp",
            sizes: "32x32",
            type: "image/webp",
          },
          {
            src: "/images/logo/favicon-96x96.webp",
            sizes: "96x96",
            type: "image/webp",
          },
          {
            src: "/images/logo/favicon-256x256.webp",
            sizes: "256x256",
            type: "image/webp",
          },
          {
            src: "/images/logo/favicon-512x512.webp",
            sizes: "512x512",
            type: "image/webp",
          },
        ],
      },
    },
  }),
);
