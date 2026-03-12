const normalizePath = (input = "") => String(input).replace(/\\/g, "/");

const stripExtension = (input = "") => String(input).replace(/\.[^.]+$/, "");

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

export const getPostPublicPath = (post) => post?.permalink || post?.regularPath || "/";
