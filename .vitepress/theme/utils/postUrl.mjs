import { generateId } from "./commonTools.mjs";

const normalizePath = (input = "") => String(input).replace(/\\/g, "/");

const stripExtension = (input = "") => String(input).replace(/\.[^.]+$/, "");
const stripLeadingSlash = (input = "") => String(input).replace(/^\/+/, "");

const toArray = (input) => {
  if (Array.isArray(input)) return input;
  if (typeof input === "string" && input.trim()) {
    return input.split(/[,\s]+/).filter(Boolean);
  }
  return [];
};

const buildSlug = ({ topic, tags, relativePath }) => {
  const topicList = toArray(topic);
  const tagList = toArray(tags);

  if (topicList.length > 0) {
    return topicList.join("-").toLowerCase();
  }

  if (tagList.length > 0) {
    const slugFromTags = tagList
      .filter((tag) => /^[a-zA-Z0-9-]+$/.test(tag))
      .slice(0, 4)
      .join("-")
      .toLowerCase();

    if (slugFromTags) return slugFromTags;
  }

  const normalized = normalizePath(relativePath);
  const filename = stripExtension(normalized.split("/").pop() || "");
  return filename
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "");
};

export const buildPostUrlData = ({ relativePath, date, topic, tags }) => {
  if (!relativePath || !date) return null;

  const parsedDate = new Date(date);
  if (Number.isNaN(parsedDate.getTime())) return null;

  const normalizedRelativePath = normalizePath(relativePath);
  const slug = buildSlug({ topic, tags, relativePath: normalizedRelativePath });
  if (!slug) return null;

  const year = parsedDate.getFullYear();
  const month = String(parsedDate.getMonth() + 1).padStart(2, "0");
  const day = String(parsedDate.getDate()).padStart(2, "0");
  const dateStr = `${year}${month}${day}`;
  const legacyPath = `/${stripExtension(normalizedRelativePath)}.html`;
  const legacyCleanPath = legacyPath.replace(/\.html$/, "");
  const permalink = `/posts/${year}/${month}/${day}/${slug}`;

  return {
    slug,
    id: `${dateStr}-${slug}`,
    legacyPath,
    legacyCleanPath,
    permalink,
  };
};

export const buildPageViewPathCandidates = ({ permalink, legacyPath, legacyCleanPath }) => {
  const resolvedLegacyCleanPath =
    legacyCleanPath || (typeof legacyPath === "string" ? legacyPath.replace(/\.html$/, "") : "");
  const candidates = [permalink, resolvedLegacyCleanPath, legacyPath];
  return [...new Set(candidates.filter((item) => typeof item === "string" && item.length > 0))];
};

export const buildSourcePathFromLegacyPath = (legacyPath = "") =>
  stripLeadingSlash(normalizePath(String(legacyPath)).replace(/\.html$/, ".md"));

export const buildRewritePathFromPermalink = (permalink = "") =>
  `${stripLeadingSlash(normalizePath(String(permalink)))}.md`;

export const buildPostRewriteRules = (posts = []) => {
  const rewrites = {};
  const seenTargets = new Map();

  posts.forEach((post) => {
    if (!post?.legacyPath || !post?.permalink) return;

    const sourcePath = buildSourcePathFromLegacyPath(post.legacyPath);
    const targetPath = buildRewritePathFromPermalink(post.permalink);
    if (!sourcePath || !targetPath) return;

    const existingSource = seenTargets.get(targetPath);
    if (existingSource && existingSource !== sourcePath) {
      throw new Error(
        `[post-url] Duplicate permalink target "${targetPath}" for "${existingSource}" and "${sourcePath}"`,
      );
    }

    seenTargets.set(targetPath, sourcePath);
    rewrites[sourcePath] = targetPath;
  });

  return rewrites;
};

export const buildLegacyRedirectHtml = (targetPath = "/") => {
  const safeTarget = String(targetPath || "/");
  const escapedTarget = safeTarget.replace(/&/g, "&amp;").replace(/"/g, "&quot;");

  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <meta http-equiv="refresh" content="0; url=${escapedTarget}">
    <link rel="canonical" href="${escapedTarget}">
    <script>location.replace(${JSON.stringify(safeTarget)});</script>
  </head>
  <body>
    <p>Redirecting to <a href="${escapedTarget}">${escapedTarget}</a></p>
  </body>
</html>
`;
};

export const resolveCurrentPostId = (pageData = {}, frontmatter = {}) => {
  if (frontmatter?.id !== undefined && frontmatter?.id !== null && frontmatter.id !== "") {
    return frontmatter.id;
  }

  const fallbackPath = pageData?.filePath || pageData?.relativePath;
  if (!fallbackPath) return null;
  return generateId(fallbackPath);
};

export const getPostPublicPath = (post) => post?.permalink || post?.regularPath || "/";
