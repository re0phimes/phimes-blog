import { generateId } from "./commonTools.mjs";
import { globby } from "globby";
import matter from "gray-matter";
import fs from "fs-extra";
import path from "node:path";
import crypto from "node:crypto";

// 封面图片缓存目录
const COVERS_DIR = path.resolve(process.cwd(), "public/covers");

/**
 * 下载外部图片到本地
 * @param {string} url - 图片 URL
 * @returns {Promise<string|null>} - 本地路径或 null
 */
const downloadCover = async (url) => {
  if (!url || !url.startsWith("http")) return url;

  try {
    // 用 URL hash 作为文件名
    const hash = crypto.createHash("md5").update(url).digest("hex").slice(0, 12);
    const ext = path.extname(new URL(url).pathname) || ".png";
    const filename = `${hash}${ext}`;
    const localPath = path.join(COVERS_DIR, filename);

    // 已存在则跳过
    if (fs.existsSync(localPath)) {
      return `/covers/${filename}`;
    }

    // 确保目录存在
    await fs.ensureDir(COVERS_DIR);

    // 下载图片
    const res = await fetch(url, { timeout: 10000 });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const buffer = Buffer.from(await res.arrayBuffer());
    await fs.writeFile(localPath, buffer);
    console.log(`[cover] Downloaded: ${filename}`);
    return `/covers/${filename}`;
  } catch (e) {
    console.warn(`[cover] Failed to download ${url}:`, e.message);
    return null; // 下载失败返回 null
  }
};

// 读取页面浏览量数据
const loadPopularData = () => {
  try {
    const popularPath = path.resolve(process.cwd(), "data/popular.json");
    if (fs.existsSync(popularPath)) {
      return fs.readJSONSync(popularPath);
    }
  } catch (e) {
    console.warn("[getPostData] Failed to load popular.json:", e.message);
  }
  return {};
};

/**
 * 获取 posts 目录下所有 Markdown 文件的路径
 * @returns {Promise<string[]>} - 文件路径数组
 */
const getPostMDFilePaths = async () => {
  try {
    // 获取所有 md 文件路径
    let paths = await globby(["**.md"], {
      ignore: ["node_modules", "pages", ".vitepress", "README.md"],
    });
    // 过滤路径，只包括 'posts' 目录下的文件
    return paths.filter((item) => item.includes("posts/"));
  } catch (error) {
    console.error("获取文章路径时出错:", error);
    throw error;
  }
};

/**
 * 基于 frontMatter 日期降序排序文章
 * @param {Object} obj1 - 第一篇文章对象
 * @param {Object} obj2 - 第二篇文章对象
 * @returns {number} - 比较结果
 */
const compareDate = (obj1, obj2) => {
  return obj1.date < obj2.date ? 1 : -1;
};
const comparePostPriority = (a, b) => {
  if (a.top && !b.top) {
    return -1;
  }
  if (!a.top && b.top) {
    return 1;
  }
  return compareDate(a, b);
};

/**
 * 获取所有文章，读取其内容并解析 front matter
 * @returns {Promise<Object[]>} - 文章对象数组
 */
export const getAllPosts = async () => {
  try {
    // 加载页面浏览量数据
    const popularData = loadPopularData();

    // 获取所有 Markdown 文件的路径
    let paths = await getPostMDFilePaths();
    // 读取和处理每个 Markdown 文件的内容
    let posts = await Promise.all(
      paths.map(async (item) => {
        try {
          // 读取文件内容
          const content = await fs.readFile(item, "utf-8");
          // 文件的元数据
          const stat = await fs.stat(item);
          // 获取文件创建时间和最后修改时间
          const { birthtimeMs, mtimeMs } = stat;
          // 解析 front matter
          const { data, content: markdownContent } = matter(content);
          const { title, date, categories, description, tags, tag, top, cover, version, lastUpdated } = data;

          // Most Popular（策展/排序字段，MVP 预留）
          // - popular: boolean
          // - popular_rank / popularRank: number（越大越靠前，具体策略在首页 selector 中定义）
          const rawPopularRank = data?.popular_rank ?? data?.popularRank;
          const popularRankNumber =
            rawPopularRank === undefined || rawPopularRank === null ? undefined : Number(rawPopularRank);

          // 从 popular.json 获取页面浏览量作为 popularRank
          const regularPath = `/${item.replace(".md", ".html")}`;
          const viewCount = popularData[regularPath] || 0;
          // 优先使用 frontmatter 中的 popularRank，否则使用浏览量
          const popularRank = Number.isFinite(popularRankNumber) ? popularRankNumber : (viewCount > 0 ? viewCount : undefined);
          const popular = Boolean(data?.popular) || popularRank !== undefined;
          
          // 从文章内容中提取第一张图片作为封面
          let articleCover = cover; // 优先使用 front matter 中的 cover
          if (!articleCover && markdownContent) {
            // 匹配 markdown 图片语法: ![alt](url) 或 ![alt](url "title")
            const imageRegex = /!\[.*?\]\((.*?)(?:\s+".*?")?\)/;
            const imageMatch = markdownContent.match(imageRegex);
            if (imageMatch && imageMatch[1]) {
              articleCover = imageMatch[1].trim();
            }
          }

          // 下载外部封面图片到本地
          if (articleCover) {
            articleCover = await downloadCover(articleCover);
          }
          
          // 如果没有描述，从文章内容中提取摘要
          let autoDescription = description;
          if (!autoDescription && markdownContent) {
            // 移除markdown语法，提取纯文本
            const plainText = markdownContent
              .replace(/^#{1,6}\s+/gm, '') // 移除标题
              .replace(/\*\*(.*?)\*\*/g, '$1') // 移除粗体
              .replace(/\*(.*?)\*/g, '$1') // 移除斜体
              .replace(/`(.*?)`/g, '$1') // 移除行内代码
              .replace(/```[\s\S]*?```/g, '') // 移除代码块
              .replace(/!\[.*?\]\(.*?\)/g, '') // 移除图片
              .replace(/\[.*?\]\(.*?\)/g, '') // 移除链接
              .replace(/\n+/g, ' ') // 替换换行为空格
              .trim();
            
            // 提取前150个字符作为摘要
            if (plainText.length > 0) {
              autoDescription = plainText.length > 150 
                ? plainText.substring(0, 150) + '...' 
                : plainText;
            }
          }
          
          // 计算文章的过期天数
          const expired = Math.floor(
            (new Date().getTime() - new Date(date).getTime()) / (1000 * 60 * 60 * 24),
          );
          // 返回文章对象
          // 兼容 tag（单数）和 tags（复数）
          const resolvedTags = tags || tag || undefined;

          return {
            id: generateId(item),
            title: title || "未命名文章",
            date: date ? new Date(date).getTime() : birthtimeMs,
            lastModified: mtimeMs,
            lastUpdated: lastUpdated ? new Date(lastUpdated).getTime() : undefined,
            version: version || undefined,
            expired,
            tags: resolvedTags,
            categories,
            description: autoDescription,
            regularPath: `/${item.replace(".md", ".html")}`,
            top,
            cover: articleCover,
            popular,
            popularRank,
          };
        } catch (error) {
          console.error(`处理文章文件 '${item}' 时出错:`, error);
          throw error;
        }
      }),
    );
    // 根据日期排序文章
    posts.sort(comparePostPriority);
    return posts;
  } catch (error) {
    console.error("获取所有文章时出错:", error);
    throw error;
  }
};

/**
 * 获取所有标签及其相关文章的统计信息
 * @param {Object[]} postData - 包含文章信息的数组
 * @returns {Object} - 包含标签统计信息的对象
 */
export const getAllType = (postData) => {
  const tagData = {};
  // 遍历数据
  postData.map((item) => {
    // 检查是否有 tags 属性
    if (!item.tags || item.tags.length === 0) return;
    // 处理标签
    if (typeof item.tags === "string") {
      // 支持逗号或空格分隔
      item.tags = item.tags.split(/[,\s]+/).filter(Boolean);
    }
    // 遍历文章的每个标签
    item.tags.forEach((tag) => {
      // 初始化标签的统计信息，如果不存在
      if (!tagData[tag]) {
        tagData[tag] = {
          count: 1,
          articles: [item],
        };
      } else {
        // 如果标签已存在，则增加计数和记录所属文章
        tagData[tag].count++;
        tagData[tag].articles.push(item);
      }
    });
  });
  return tagData;
};

/**
 * 获取所有分类及其相关文章的统计信息
 * @param {Object[]} postData - 包含文章信息的数组
 * @returns {Object} - 包含标签统计信息的对象
 */
export const getAllCategories = (postData) => {
  const catData = {};
  // 遍历数据
  postData.map((item) => {
    if (!item.categories || item.categories.length === 0) return;
    // 处理标签
    if (typeof item.categories === "string") {
      // 以逗号分隔
      item.categories = item.categories.split(",");
    }
    // 遍历文章的每个标签
    item.categories.forEach((tag) => {
      // 初始化标签的统计信息，如果不存在
      if (!catData[tag]) {
        catData[tag] = {
          count: 1,
          articles: [item],
        };
      } else {
        // 如果标签已存在，则增加计数和记录所属文章
        catData[tag].count++;
        catData[tag].articles.push(item);
      }
    });
  });
  return catData;
};

/**
 * 获取所有年份及其相关文章的统计信息
 * @param {Object[]} postData - 包含文章信息的数组
 * @returns {Object} - 包含归档统计信息的对象
 */
export const getAllArchives = (postData) => {
  const archiveData = {};
  // 遍历数据
  postData.forEach((item) => {
    // 检查是否有 date 属性
    if (item.date) {
      // 将时间戳转换为日期对象
      const date = new Date(item.date);
      // 获取年份
      const year = date.getFullYear().toString();
      // 初始化该年份的统计信息，如果不存在
      if (!archiveData[year]) {
        archiveData[year] = {
          count: 1,
          articles: [item],
        };
      } else {
        // 如果年份已存在，则增加计数和记录所属文章
        archiveData[year].count++;
        archiveData[year].articles.push(item);
      }
    }
  });
  // 提取年份并按降序排序
  const sortedYears = Object.keys(archiveData).sort((a, b) => parseInt(b) - parseInt(a));
  return { data: archiveData, year: sortedYears };
};
