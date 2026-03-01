# URL 结构与 Sitemap 优化设计

**日期**：2026-03-01
**作者**：Phimes
**状态**：已确认

## 1. 背景与目标

### 当前问题

1. **URL 结构不友好**：使用完整中文标题作为 URL，导致 URL 过长且不利于 SEO
   - 示例：`/posts/2026/KV%20Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。`

2. **Sitemap 信息不完整**：现有 sitemap.xml 只包含 `<loc>` 和 `<lastmod>`，缺少 SEO 优化的关键元素
   - 缺少 `<priority>` - 页面优先级
   - 缺少 `<changefreq>` - 更新频率

### 优化目标

1. ✅ 保留中文文件名（便于内容管理）
2. ✅ 生成 SEO 友好的英文 URL（日期 + 关键词）
3. ✅ 为每篇文章生成唯一 ID（便于引用和短链接）
4. ✅ 优化 sitemap，添加 priority 和 changefreq
5. ✅ 不破坏现有链接（旧 URL 仍可访问）

## 2. URL 结构优化设计

### 目标 URL 格式

```
https://blog.phimes.top/posts/{年份}/{月份}/{日期}/{关键词-slug}

示例：
- https://blog.phimes.top/posts/2026/01/16/kv-cache-mqa-gqa-mla
- https://blog.phimes.top/posts/2025/06/16/vue-syntax-summary
```

### 文件结构

**保留中文文件名**：
```
posts/2026/KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解。.md
```

**Frontmatter 配置**：
```yaml
---
title: KV Cache（二）：从如何让GPU不摸鱼开始思考——MQA、GQA到MLA的计算拆解
id: 20260116-kv-cache-mqa-gqa-mla  # 唯一 ID
date: 2026-01-16
topic: [transformer, kv-cache, attention, roofline]
permalink: /posts/2026/01/16/kv-cache-mqa-gqa-mla  # 自动生成
---
```

### 实现逻辑

在 `.vitepress/config.mjs` 的 `transformPageData` 钩子中实现：

```javascript
transformPageData: async (pageData) => {
  // 只处理 posts 目录下的文章
  if (!pageData.relativePath.startsWith('posts/')) return;

  const frontmatter = pageData.frontmatter;

  // 1. 提取日期（必需）
  if (!frontmatter.date) return;
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
```

### Slug 生成示例

| 场景 | 输入 | 输出 slug |
|------|------|-----------|
| 有 topic | `topic: [transformer, kv-cache, attention]` | `transformer-kv-cache-attention` |
| 只有 tags | `tags: [大模型, KV-Cache, MQA, GQA]` | `kv-cache-mqa-gqa` |
| 都没有 | 文件名: `KV Cache（二）.md` | `kv-cache-二` |

### 唯一 ID 格式

```
{年月日}-{slug}

示例：
- 20260116-kv-cache-mqa-gqa-mla
- 20250616-vue-syntax-summary
```

**用途**：
1. 内部引用和数据库主键
2. 短链接服务（如 `/p/20260116-kv-cache-mqa-gqa-mla`）
3. API 查询
4. 评论系统绑定

## 3. Sitemap 优化设计

### 优化目标

为 sitemap.xml 中的每个 URL 添加：
- `<priority>` - 页面优先级（0.0-1.0）
- `<changefreq>` - 更新频率

### 优先级和更新频率规则

| 页面类型 | Priority | Changefreq | 说明 |
|---------|----------|------------|------|
| 首页 | 1.0 | weekly | 每周发布新文章 |
| 热门文章 | 0.8 | monthly | 有 popularRank 的文章 |
| 最近 3 个月文章 | 0.7 | monthly | 新文章可能会更新 |
| 旧文章 | 0.6 | yearly | 基本不再修改 |
| 分类页 | 0.5 | weekly | 新文章会影响分类 |
| 标签页 | 0.5 | weekly | 新文章会影响标签 |
| 归档页 | 0.4 | monthly | 定期归档 |
| 静态页面 | 0.5 | yearly | 关于、友链等 |

### 实现方式

在 `.vitepress/config.mjs` 的 `sitemap` 配置中添加 `transformItems` 钩子：

```javascript
export default defineConfig({
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
  }
});
```

### 生成效果

```xml
<!-- 首页 -->
<url>
  <loc>https://blog.phimes.top/</loc>
  <lastmod>2026-03-01T00:00:00.000Z</lastmod>
  <changefreq>weekly</changefreq>
  <priority>1.0</priority>
</url>

<!-- 热门文章 -->
<url>
  <loc>https://blog.phimes.top/posts/2026/01/16/kv-cache-mqa-gqa-mla</loc>
  <lastmod>2026-02-27T00:00:00.000Z</lastmod>
  <changefreq>monthly</changefreq>
  <priority>0.8</priority>
</url>

<!-- 旧文章 -->
<url>
  <loc>https://blog.phimes.top/posts/2024/06/16/vue-syntax-summary</loc>
  <lastmod>2025-06-16T12:13:04.000Z</lastmod>
  <changefreq>yearly</changefreq>
  <priority>0.6</priority>
</url>
```

## 4. 实施计划

### 第一阶段：URL 结构优化

1. 修改 `.vitepress/config.mjs`，添加 `transformPageData` 钩子
2. 实现 slug 生成逻辑（topic → tags → filename）
3. 实现唯一 ID 生成逻辑
4. 测试新 URL 是否正常工作
5. 验证旧 URL 是否仍可访问

### 第二阶段：Sitemap 优化

1. 修改 `.vitepress/config.mjs`，添加 `sitemap.transformItems` 钩子
2. 实现页面分类逻辑
3. 为每个页面类型设置 priority 和 changefreq
4. 构建并验证生成的 sitemap.xml
5. 提交到 Google Search Console 和 Bing Webmaster

### 第三阶段：重定向配置（可选）

如果发现旧 URL 有外部链接或搜索引擎索引：
1. 在构建时生成重定向映射
2. 配置 Vercel 重定向规则
3. 或使用 VitePress 的 alias 功能

## 5. 预期效果

### URL 优化效果

- ✅ URL 简洁易读：`/posts/2026/01/16/kv-cache-mqa-gqa-mla`
- ✅ 包含关键词，利于 SEO
- ✅ 日期结构清晰，便于归档
- ✅ 唯一 ID 便于引用和短链接

### Sitemap 优化效果

- ✅ 搜索引擎能识别页面优先级
- ✅ 爬虫根据 changefreq 合理分配抓取频率
- ✅ 热门内容优先被索引
- ✅ 节省爬虫预算，提高抓取效率

### SEO 提升

1. **关键词优化**：URL 包含核心关键词
2. **爬虫效率**：优先抓取重要页面
3. **索引速度**：新文章更快被索引
4. **用户体验**：URL 可读性强，便于分享

## 6. 风险与注意事项

### 风险

1. **URL 变更风险**：可能影响已有的外部链接和搜索引擎索引
2. **Slug 冲突**：同一天发布相同主题的文章可能导致 URL 冲突

### 缓解措施

1. **保留旧 URL**：VitePress 会自动处理文件名到 URL 的映射，旧 URL 仍可访问
2. **重定向配置**：如有需要，可配置 301 重定向
3. **冲突检测**：在生成 ID 时检测冲突，自动添加序号后缀

### 注意事项

1. **changefreq 诚实原则**：应反映真实更新频率，不要夸大
2. **priority 相对性**：是网站内的相对优先级，不是绝对值
3. **渐进式迁移**：新文章使用新 URL，旧文章可保持不变

## 7. 后续优化

1. **短链接服务**：基于唯一 ID 实现短链接（如 `/p/{id}`）
2. **Analytics 集成**：根据真实访问数据动态调整 priority
3. **自动化测试**：验证 URL 生成和 sitemap 正确性
4. **监控与分析**：跟踪 SEO 效果，持续优化

---

**设计确认**：✅ 已确认
**下一步**：创建实施计划
