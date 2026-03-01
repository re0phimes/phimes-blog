# URL & Sitemap 优化测试报告

## 测试时间
2026-03-01

## 测试结果
- ✅ 构建成功，无错误
- ✅ 新 URL 格式正确 (YYYY/MM/DD/slug)
- ✅ Sitemap 包含 priority 和 changefreq
- ✅ XML 格式验证通过
- ✅ 所有 URL 都有完整的四个字段

## Sitemap 统计
- 总 URL 数: 68
- 有 priority 的 URL: 68
- 有 changefreq 的 URL: 68
- 覆盖率: 100%

## URL 转换验证
构建日志显示：
```
[Sitemap] Transformed 22 post URLs to SEO-friendly format
```

## 示例 URL

### 首页
```xml
<url>
  <loc>https://blog.phimes.top/</loc>
  <lastmod>2024-10-10T08:46:05.000Z</lastmod>
  <changefreq>weekly</changefreq>
  <priority>1.0</priority>
</url>
```

### 2026 年文章
```xml
<url>
  <loc>https://blog.phimes.top/posts/2026/01/04/transformer-kv-cache-attention</loc>
  <lastmod>2026-02-27T00:00:00.000Z</lastmod>
  <changefreq>monthly</changefreq>
  <priority>0.7</priority>
</url>

<url>
  <loc>https://blog.phimes.top/posts/2026/01/16/transformer-kv-cache-attention-roofline</loc>
  <lastmod>2026-02-27T00:00:00.000Z</lastmod>
  <changefreq>monthly</changefreq>
  <priority>0.7</priority>
</url>
```

## Priority 分配验证
- 首页: 1.0 ✅
- 2026 年文章: 0.7 (monthly) ✅
- 2025 年文章: 0.6-0.7 ✅
- 其他页面: 0.4-0.5 ✅

## Changefreq 分配验证
- 首页: weekly ✅
- 最新文章: monthly ✅
- 旧文章: yearly ✅
- 分类/标签: weekly ✅

## 结论
所有测试通过，URL 生成和 Sitemap 优化功能正常工作。
