import { defineConfig } from "vitepress";
import { createRssFile } from "./theme/utils/generateRSS.mjs";
import { createBlogIndex } from "./theme/utils/generateBlogIndex.mjs";
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
            transformItems: (items) => {
                // 为每个 post 生成 SEO 友好的 URL 并添加 SEO 字段
                let transformedCount = 0;
                const now = new Date();
                const threeMonthsAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);

                return items.map(item => {
                    // 1. 首页
                    if (item.url === '/' || item.url === '') {
                        return { ...item, priority: 1.0, changefreq: 'weekly' };
                    }

                    // 2. 文章页
                    if (item.url && item.url.startsWith('posts/')) {
                        // 尝试从 postData 中找到对应的文章数据
                        const post = postData.find(p => {
                            // item.url 格式: "posts/2024/vue语法总结"
                            // p.regularPath 格式: "/posts/2024/vue语法总结.html"
                            const postPath = p.regularPath.replace(/^\//, '').replace(/\.html$/, '');
                            return postPath === item.url;
                        });

                        if (post && post.date) {
                            const date = new Date(post.date);
                            const year = date.getFullYear();
                            const month = String(date.getMonth() + 1).padStart(2, '0');
                            const day = String(date.getDate()).padStart(2, '0');

                            // 生成 slug（智能 fallback）
                            let slug = '';

                            // 优先级 1: topic 字段
                            if (post.topic && post.topic.length > 0) {
                                slug = post.topic.join('-');
                            }
                            // 优先级 2: tags 字段（过滤中文，保留英文/缩写）
                            else if (post.tags && post.tags.length > 0) {
                                slug = post.tags
                                    .filter(tag => /^[a-zA-Z0-9-]+$/.test(tag))
                                    .slice(0, 4)
                                    .join('-')
                                    .toLowerCase();
                            }
                            // 优先级 3: 从文件名提取（fallback）
                            if (!slug) {
                                const filename = post.regularPath.split('/').pop().replace('.html', '');
                                slug = filename
                                    .toLowerCase()
                                    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
                                    .replace(/^-+|-+$/g, '');
                            }

                            // 生成新的 URL
                            if (slug) {
                                item.url = `posts/${year}/${month}/${day}/${slug}`;
                                transformedCount++;
                            }

                            // 添加 SEO 字段
                            // 热门文章
                            if (post.popular || post.popularRank) {
                                return { ...item, priority: 0.8, changefreq: 'monthly' };
                            }

                            // 最近 3 个月的文章
                            const postDate = new Date(post.date);
                            if (postDate > threeMonthsAgo) {
                                return { ...item, priority: 0.7, changefreq: 'monthly' };
                            }

                            // 旧文章
                            return { ...item, priority: 0.6, changefreq: 'yearly' };
                        }
                    }

                    // 3. 分类页
                    if (item.url.startsWith('pages/categories/')) {
                        return { ...item, priority: 0.5, changefreq: 'weekly' };
                    }

                    // 4. 标签页
                    if (item.url.startsWith('pages/tags/')) {
                        return { ...item, priority: 0.5, changefreq: 'weekly' };
                    }

                    // 5. 归档页
                    if (item.url.includes('/archives')) {
                        return { ...item, priority: 0.4, changefreq: 'monthly' };
                    }

                    // 6. 其他页面
                    return { ...item, priority: 0.5, changefreq: 'yearly' };
                }).filter(item => {
                    // 打印转换统计
                    if (items.indexOf(item) === items.length - 1) {
                        console.log(`[Sitemap] Transformed ${transformedCount} post URLs to SEO-friendly format`);
                    }
                    return true;
                });
            }
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
            // 新增：URL 结构优化
            // 只处理 posts 目录下的文章
            let customUrl = null;
            if (pageData.relativePath.startsWith('posts/')) {
                const frontmatter = pageData.frontmatter;

                // 1. 提取日期（必需）
                if (frontmatter.date) {
                    const date = new Date(frontmatter.date);
                    const year = date.getFullYear();
                    const month = String(date.getMonth() + 1).padStart(2, '0');
                    const day = String(date.getDate()).padStart(2, '0');
                    const dateStr = `${year}${month}${day}`;

                    // 2. 生成 slug（智能 fallback）
                    let slug = '';

                    // 优先级 1: topic 字段
                    if (frontmatter.topic && frontmatter.topic.length > 0) {
                        slug = frontmatter.topic.join('-');
                    }
                    // 优先级 2: tags 字段（过滤中文，保留英文/缩写）
                    else if (frontmatter.tags && frontmatter.tags.length > 0) {
                        slug = frontmatter.tags
                            .filter(tag => /^[a-zA-Z0-9-]+$/.test(tag))
                            .slice(0, 4)
                            .join('-')
                            .toLowerCase();
                    }
                    // 优先级 3: 从文件名提取（fallback）
                    else {
                        const filename = pageData.relativePath.split('/').pop().replace('.md', '');
                        slug = filename
                            .toLowerCase()
                            .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
                            .replace(/^-+|-+$/g, '');
                    }

                    // 3. 生成唯一 ID
                    frontmatter.id = `${dateStr}-${slug}`;

                    // 4. 生成 permalink（用于 canonical URL）
                    customUrl = `/posts/${year}/${month}/${day}/${slug}`;
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
        },
        // transformHtml
        transformHtml: (html) => {
            return jumpRedirect(html, themeConfig);
        },
        // buildEnd
        buildEnd: async (config) => {
            await createRssFile(config, themeConfig);
            await createBlogIndex(config, themeConfig);
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
