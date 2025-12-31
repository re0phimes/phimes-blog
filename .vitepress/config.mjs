import { defineConfig } from "vitepress";
import { createRssFile } from "./theme/utils/generateRSS.mjs";
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

// 获取全局数据
const postData = await getAllPosts();

// 获取主题配置
const themeConfig = await getThemeConfig();

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
        if (!post || typeof post.regularPath !== "string") return;
        const score = scoreByPath.get(post.regularPath);
        if (!Number.isFinite(score)) return;
        post.popularRank = score;
        post.popular = true;
    });
};

await mergeMostPopularExternalCache(postData, themeConfig);

// https://vitepress.dev/reference/site-config
export default withPwa(
    defineConfig({
        title: themeConfig.siteMeta.title,
        description: themeConfig.siteMeta.description,
        lang: themeConfig.siteMeta.lang,
        // 简洁的 URL
        cleanUrls: true,
        // 最后更新时间戳
        lastUpdated: true,
        // 主题
        appearance: "dark",
        head: [
            // 1. 使用展开语法(...)保留所有原有的 head 标签
            // 这行代码至关重要，确保不会破坏您主题原有的功能和样式
            ...themeConfig.inject.header,

            // 2. 添加 Microsoft Clarity 的跟踪脚本
            [
                'script',
                {}, // 这个空对象用于存放 script 标签的属性，这里我们不需要额外属性
                // 下面是您从 Clarity 官网获取的脚本内容，已包含您的 Project ID
                `(function(c,l,a,r,i,t,y){
          c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
          t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
          y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
      })(window, document, "clarity", "script", "ty1lrwm16k");`
            ]
        ],

        // sitemap
        sitemap: {
            hostname: themeConfig.siteMeta.site,
        },
        // 主题配置
        themeConfig: {
            ...themeConfig,
            // 必要数据
            postData: postData,
            tagsData: getAllType(postData),
            categoriesData: getAllCategories(postData),
            archivesData: getAllArchives(postData),
        },
        // markdown
        markdown: {
            math: true,
            lineNumbers: true,
            toc: { level: [1, 2, 3] },
            image: {
                lazyLoading: true,
            },
            config: (md) => markdownConfig(md, themeConfig),
        },
        // 构建排除
        srcExclude: ["**/README.md", "**/TODO.md"],
        // transformHead
        transformPageData: async (pageData) => {
            // canonical URL
            const canonicalUrl = `${themeConfig.siteMeta.site}/${pageData.relativePath}`
                .replace(/index\.md$/, "")
                .replace(/\.md$/, "");
            pageData.frontmatter.head ??= [];
            pageData.frontmatter.head.push(["link", { rel: "canonical", href: canonicalUrl }]);
        },
        // transformHtml
        transformHtml: (html) => {
            return jumpRedirect(html, themeConfig);
        },
        // buildEnd
        buildEnd: async (config) => {
            await createRssFile(config, themeConfig);
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
