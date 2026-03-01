# URL 结构与 Sitemap 优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化博客 URL 结构（日期+关键词）并增强 sitemap SEO 信息（priority + changefreq）

**Architecture:** 在 VitePress 的 transformPageData 钩子中自动生成 permalink 和唯一 ID，在 sitemap.transformItems 中为不同页面类型设置 SEO 参数

**Tech Stack:** VitePress 1.6.3, Node.js, JavaScript

---

## Task 1: 实现 URL 生成逻辑

**Files:**
- Modify: `.vitepress/config.mjs:158-174` (transformPageData 部分)

**Step 1: 备份当前配置**

```bash
cp .vitepress/config.mjs .vitepress/config.mjs.backup
```

**Step 2: 在 transformPageData 中添加 URL 生成逻辑**

在 `.vitepress/config.mjs` 的 `transformPageData` 函数中，添加以下代码（在 canonical URL 逻辑之后）：

```javascript
transformPageData: async (pageData) => {
    // 现有的 canonical URL 逻辑
    const canonicalUrl = `${themeConfig.siteMeta.site}/${pageData.relativePath}`
        .replace(/index\.md$/, "")
        .replace(/\.md$/, "");
    pageData.frontmatter.head ??= [];
    pageData.frontmatter.head.push(["link", { rel: "canonical", href: canonicalUrl }]);

    // 新增：URL 结构优化
    // 只处理 posts 目录下的文章
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

            // 4. 生成 permalink
            frontmatter.permalink = `/posts/${year}/${month}/${day}/${slug}`;
        }
    }
},
```

**Step 3: 保存文件**

保存 `.vitepress/config.mjs`

**Step 4: 测试构建**

```bash
npm run build
```

Expected: 构建成功，无错误

**Step 5: 检查生成的 URL**

```bash
# 查看生成的 sitemap，检查新 URL 格式
cat .vitepress/dist/sitemap.xml | grep -A 3 "posts/2026"
```

Expected: 看到类似 `/posts/2026/01/16/kv-cache-mqa-gqa-mla` 的 URL

**Step 6: 启动开发服务器测试**

```bash
npm run dev
```

访问新 URL，验证页面能正常访问

**Step 7: 验证旧 URL 仍可访问**

访问旧的中文 URL（如 `/posts/2026/KV Cache（二）...`），确认仍能访问

**Step 8: 提交更改**

```bash
git add .vitepress/config.mjs
git commit -m "feat: 实现 SEO 友好的 URL 结构（日期+关键词）

- 在 transformPageData 中自动生成 permalink
- 支持 topic/tags/filename 三级 fallback
- 为每篇文章生成唯一 ID (YYYYMMDD-slug)
- 保留旧 URL 兼容性"
```

---

## Task 2: 实现 Sitemap 优化逻辑

**Files:**
- Modify: `.vitepress/config.mjs:132-135` (sitemap 配置部分)

**Step 1: 修改 sitemap 配置，添加 transformItems 钩子**

将现有的简单 sitemap 配置：

```javascript
// sitemap
sitemap: {
    hostname: themeConfig.siteMeta.site,
},
```

替换为：

```javascript
// sitemap
sitemap: {
    hostname: themeConfig.siteMeta.site,
    transformItems: (items) => {
        const now = new Date();
        const threeMonthsAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);

        return items.map(item => {
            // 1. 首页
            if (item.url === '/') {
                return { ...item, priority: 1.0, changefreq: 'weekly' };
            }

            // 2. 文章页
            if (item.url.startsWith('/posts/')) {
                const post = postData.find(p => p.regularPath === item.url);

                if (post) {
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
            if (item.url.startsWith('/pages/categories/')) {
                return { ...item, priority: 0.5, changefreq: 'weekly' };
            }

            // 4. 标签页
            if (item.url.startsWith('/pages/tags/')) {
                return { ...item, priority: 0.5, changefreq: 'weekly' };
            }

            // 5. 归档页
            if (item.url.includes('/archives')) {
                return { ...item, priority: 0.4, changefreq: 'monthly' };
            }

            // 6. 其他页面
            return { ...item, priority: 0.5, changefreq: 'yearly' };
        });
    }
},
```

**Step 2: 保存文件**

保存 `.vitepress/config.mjs`

**Step 3: 重新构建**

```bash
npm run build
```

Expected: 构建成功

**Step 4: 验证 sitemap 内容**

```bash
# 检查首页的 priority 和 changefreq
cat .vitepress/dist/sitemap.xml | grep -A 4 '<loc>https://blog.phimes.top/</loc>'
```

Expected: 看到 `<priority>1.0</priority>` 和 `<changefreq>weekly</changefreq>`

**Step 5: 检查文章页的 SEO 信息**

```bash
# 检查热门文章
cat .vitepress/dist/sitemap.xml | grep -A 4 'posts/2026' | head -20
```

Expected: 看到不同文章有不同的 priority 和 changefreq

**Step 6: 验证完整性**

```bash
# 统计 sitemap 中的 URL 数量
grep -c '<url>' .vitepress/dist/sitemap.xml
```

Expected: 数量与之前一致（约 60+）

**Step 7: 提交更改**

```bash
git add .vitepress/config.mjs
git commit -m "feat: 优化 sitemap SEO 信息

- 为不同页面类型设置 priority 和 changefreq
- 首页: priority=1.0, changefreq=weekly
- 热门文章: priority=0.8, changefreq=monthly
- 最近文章: priority=0.7, changefreq=monthly
- 旧文章: priority=0.6, changefreq=yearly
- 分类/标签页: priority=0.5, changefreq=weekly"
```

---

## Task 3: 端到端测试

**Files:**
- Test: 手动测试

**Step 1: 完整构建测试**

```bash
npm run build
```

Expected: 构建成功，无警告或错误

**Step 2: 测试新 URL 访问**

启动预览服务器：

```bash
npm run preview
```

访问以下 URL 并验证：
1. 新 URL：`http://localhost:4173/posts/2026/01/16/kv-cache-mqa-gqa-mla`
2. 旧 URL：`http://localhost:4173/posts/2026/KV Cache（二）...`（URL 编码后）

Expected: 两个 URL 都能正常访问同一篇文章

**Step 3: 验证 sitemap.xml 格式**

```bash
# 检查 XML 格式是否正确
xmllint --noout .vitepress/dist/sitemap.xml 2>&1
```

如果没有 xmllint，可以在线验证：
1. 复制 `.vitepress/dist/sitemap.xml` 内容
2. 访问 https://www.xml-sitemaps.com/validate-xml-sitemap.html
3. 粘贴内容验证

Expected: XML 格式正确，无错误

**Step 4: 检查示例文章的完整信息**

```bash
# 查看一篇文章的完整 sitemap 条目
cat .vitepress/dist/sitemap.xml | grep -A 5 'kv-cache'
```

Expected: 包含 loc, lastmod, changefreq, priority 四个字段

**Step 5: 验证 ID 生成**

在浏览器开发者工具中，检查文章页面的 frontmatter 数据，确认 `id` 字段已生成

**Step 6: 记录测试结果**

创建测试报告：

```bash
echo "# URL & Sitemap 优化测试报告

## 测试时间
$(date)

## 测试结果
- ✅ 新 URL 格式正确
- ✅ 旧 URL 仍可访问
- ✅ Sitemap 包含 priority 和 changefreq
- ✅ XML 格式验证通过
- ✅ 唯一 ID 正常生成

## Sitemap 统计
- 总 URL 数: $(grep -c '<url>' .vitepress/dist/sitemap.xml)
- 首页 priority: 1.0
- 文章页 priority: 0.6-0.8
- 其他页面 priority: 0.4-0.5
" > docs/plans/2026-03-01-test-report.md
```

**Step 7: 提交测试报告**

```bash
git add docs/plans/2026-03-01-test-report.md
git commit -m "docs: 添加 URL 和 Sitemap 优化测试报告"
```

---

## Task 4: 部署和验证

**Files:**
- Deploy: Vercel

**Step 1: 推送到远程仓库**

```bash
git push origin master
```

Expected: 推送成功

**Step 2: 等待 Vercel 自动部署**

访问 Vercel Dashboard，等待部署完成

**Step 3: 验证生产环境的新 URL**

访问：`https://blog.phimes.top/posts/2026/01/16/kv-cache-mqa-gqa-mla`

Expected: 页面正常显示

**Step 4: 验证生产环境的 sitemap**

访问：`https://blog.phimes.top/sitemap.xml`

检查是否包含 priority 和 changefreq

**Step 5: 提交 sitemap 到搜索引擎**

1. **Google Search Console**
   - 访问 https://search.google.com/search-console
   - 选择你的网站
   - 左侧菜单 → Sitemaps
   - 输入 `sitemap.xml`
   - 点击"提交"

2. **Bing Webmaster Tools**
   - 访问 https://www.bing.com/webmasters
   - 选择你的网站
   - 左侧菜单 → Sitemaps
   - 输入 `https://blog.phimes.top/sitemap.xml`
   - 点击"提交"

**Step 6: 记录部署信息**

```bash
echo "# 部署记录

## 部署时间
$(date)

## 部署环境
- 平台: Vercel
- 分支: master
- Commit: $(git rev-parse HEAD)

## 验证结果
- ✅ 新 URL 在生产环境正常工作
- ✅ Sitemap 已更新
- ✅ 已提交到 Google Search Console
- ✅ 已提交到 Bing Webmaster Tools

## 后续监控
- 7 天后检查 Google Search Console 的索引状态
- 14 天后检查新 URL 的搜索排名
- 30 天后评估 SEO 效果
" >> docs/plans/2026-03-01-test-report.md
```

**Step 7: 最终提交**

```bash
git add docs/plans/2026-03-01-test-report.md
git commit -m "docs: 更新部署记录"
git push origin master
```

---

## 完成检查清单

- [ ] Task 1: URL 生成逻辑已实现
- [ ] Task 2: Sitemap 优化逻辑已实现
- [ ] Task 3: 端到端测试通过
- [ ] Task 4: 已部署到生产环境
- [ ] Sitemap 已提交到 Google Search Console
- [ ] Sitemap 已提交到 Bing Webmaster Tools
- [ ] 测试报告已创建
- [ ] 所有更改已提交到 git

---

## 后续优化建议

1. **监控 SEO 效果**（7-30 天后）
   - 检查 Google Search Console 的索引状态
   - 分析新 URL 的搜索排名变化
   - 评估爬虫抓取频率是否提升

2. **可选：实现短链接服务**
   - 基于唯一 ID 实现 `/p/{id}` 短链接
   - 配置重定向规则

3. **可选：添加重定向配置**
   - 如果发现旧 URL 有大量外部链接
   - 配置 301 重定向到新 URL

4. **持续优化**
   - 根据真实访问数据调整 priority
   - 根据更新频率调整 changefreq
