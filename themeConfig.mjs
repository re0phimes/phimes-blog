// 主题配置
export const themeConfig = {
  // 站点信息
  siteMeta: {
    // 站点标题
    title: "Phimes",
    // 站点描述
    description: "Phimes的个人博客",
    // 站点logo
    logo: "/images/logo/logo.webp",
    // 站点地址
    site: "https://blog.phimes.top",
    // 语言
    lang: "zh-CN",
    // 作者
    author: {
      name: "phimes",
      cover: "/images/logo/logo.webp",
      email: "re0phimes@outlook.com",
      link: "https://blog.phimes.top",
    },
  },
  // 备案信息
  icp: "萌ICP备114514号",
  // 建站日期
  since: "2020-07-28",
  // 每页文章数据
  postSize: 8,
  // 首页（Home）
  home: {
    // 首屏头条文章
    hero: {
      // 关掉后首页直接进列表
      enable: true,
      // 指定头条：支持 permalink / id / 文件名 / 标题；留空 = 最新一篇
      post: "",
      // 头条上方的小标签
      label: "最新",
      // 手写摘要（推荐）。留空会退回构建时截取的正文开头，读起来不像摘要。
      excerpt:
        "完整对比 MHA、MQA、GQA、MLA 四种 Attention 在 KV Cache 优化上的表现，每一步都用具体数字算清楚：每种方案到底能省多少？代价是什么？瓶颈卡在哪？",
    },
    // 头条下方的封面次条
    featured: {
      // 关掉后只剩头条 + 列表
      enable: true,
      title: "最新文章",
      // 展示条数
      limit: 3,
      // 仅首页第一页展示（/，不含 /page/2+；不含分类/标签页）
      onlyFirstPage: true,
      // 查看全部
      moreLink: "/pages/archives",
    },
  },
  // inject
  inject: {
    // 头部
    // https://vitepress.dev/zh/reference/site-config#head
    header: [
      // favicon
      ["link", { rel: "icon", href: "/favicon.ico" }],
      // RSS
      [
        "link",
        {
          rel: "alternate",
          type: "application/rss+xml",
          title: "RSS",
          href: "https://blog.phimes.top/rss.xml",
        },
      ],
      // 预载 CDN
      // iconfont
      [
        "link",
        {
          crossorigin: "anonymous",
          rel: "stylesheet",
          href: "https://cdn2.codesign.qq.com/icons/g5ZpEgx3z4VO6j2/latest/iconfont.css",
        },
      ],
      // 预载 DocSearch
      [
        "link",
        {
          href: "https://X5EBEZB53I-dsn.algolia.net",
          rel: "preconnect",
          crossorigin: "",
        },
      ],
    ],
  },
  // 导航栏菜单
  nav: [
    {
      text: "文库",
      items: [
        { text: "文章列表", link: "/pages/archives", icon: "article" },
        { text: "全部分类", link: "/pages/categories", icon: "folder" },
        { text: "全部标签", link: "/pages/tags", icon: "hashtag" },
      ],
    },
    {
      text: "可视化理解",
      link: "https://demo.phimes.top",
      icon: "code",
    },
    {
      text: "AI FAQ",
      link: "https://aifaq.phimes.top",
      icon: "chat",
    },
  ],
  // 导航栏菜单 - 左侧
  navMore: [
    {
      name: "博客",
      list: [
        {
          icon: "/images/logo/logo.webp",
          name: "主站",
          url: "https://phimes.top",
        },
      ],
    },
    {
      name: "项目",
      list: [
        {
          icon: "/images/logo/logo.webp",
          name: "Demo",
          url: "https://demo.phimes.top",
        },
      ],
    },
  ],
  // 封面配置
  cover: {
    // 是否开启双栏布局
    twoColumns: false,
    // 是否开启封面显示
    showCover: {
      // 是否开启封面显示 文章不设置cover封面会显示异常，可以设置下方默认封面
      enable: true,
      // 封面布局方式: left | right | both
      coverLayout: "left",
      // 默认封面(随机展示)
      defaultCover: [
        "https://images.unsplash.com/photo-1516116216624-53e697fedbea?w=800&h=400&fit=crop",
        "https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=800&h=400&fit=crop",
        "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&h=400&fit=crop",
        "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=800&h=400&fit=crop",
        "https://images.unsplash.com/photo-1504639725590-34d0984388bd?w=800&h=400&fit=crop",
      ],
    },
  },
  // 页脚信息
  footer: {
    // 社交链接（请确保为偶数个）
    social: [
      {
        icon: "email",
        link: "re0phimes@outlook.com",
      },
      {
        icon: "github",
        link: "https://github.com/re0phimes",
      },
    ],
    // sitemap
    // sitemap: [
    //   {
    //     text: "博客",
    //     items: [
    //       { text: "近期文章", link: "/" },
    //       { text: "全部分类", link: "/pages/categories" },
    //       { text: "全部标签", link: "/pages/tags" },
    //       // { text: "文章归档", link: "/pages/archives", newTab: true },
    //     ],
    //   },
    //   {
    //     text: "项目",
    //     items: [
    //       { text: "Home", link: "https://github.com/imsyy/home/", newTab: true },
    //       { text: "SPlayer", link: "https://github.com/imsyy/SPlayer/", newTab: true },
    //       { text: "DailyHotApi", link: "https://github.com/imsyy/DailyHotApi/", newTab: true },
    //       { text: "Snavigation", link: "https://github.com/imsyy/Snavigation/", newTab: true },
    //     ],
    //   },
    //   {
    //     text: "专栏",
    //     items: [
    //       { text: "技术分享", link: "/pages/categories/技术分享" },
    //       { text: "我的项目", link: "/pages/project" },
    //       // { text: "效率工具", link: "/pages/tools" },
    //     ],
    //   },
    //   {
    //     text: "页面",
    //     items: [
    //       { text: "畅所欲言", link: "/pages/message" },
    //       { text: "关于本站", link: "/pages/about" },
    //       { text: "隐私政策", link: "/pages/privacy" },
    //       { text: "版权协议", link: "/pages/cc" },
    //     ],
    //   },
    //   {
    //     text: "服务",
    //     items: [
    //       { text: "站点状态", link: "https://status.phimes.top/", newTab: true },
    //       { text: "一个导航", link: "https://nav.phimes.top/", newTab: true },
    //       { text: "站点订阅", link: "https://blog.phimes.top/rss.xml", newTab: true },
    //       {
    //         text: "反馈投诉",
    //         link: "https://eqnxweimkr5.feishu.cn/share/base/form/shrcnCXCPmxCKKJYI3RKUfefJre",
    //         newTab: true,
    //       },
    //     ],
    //   },
    // ],
  },
  // 评论
  comment: {
    enable: false,
    // 评论系统选择
    // artalk / twikoo
    type: "artalk",
    // artalk
    // https://artalk.js.org/
    artalk: {
      site: "",
      server: "",
    },
    // twikoo
    // https://twikoo.js.org/
    twikoo: {
      // 必填，若不想使用 CDN，可以使用 pnpm add twikoo 安装并引入
      js: "https://mirrors.sustech.edu.cn/cdnjs/ajax/libs/twikoo/1.6.39/twikoo.all.min.js",
      envId: "",
      // 环境地域，默认为 ap-shanghai，腾讯云环境填 ap-shanghai 或 ap-guangzhou；Vercel 环境不填
      region: "ap-shanghai",
      lang: "zh-CN",
    },
  },
  // 侧边栏
  aside: {
    // 目录
    toc: {
      enable: true,
    },
    // 标签
    tags: {
      enable: true,
      // 侧边栏显示多少个（首页 = 热门 Top N；文章页 = 相关标签 Top N）
      limit: 10,
    },
    // 站点数据
    siteData: {
      enable: true,
    },
  },
  // 友链
  friends: {
    // 友链朋友圈
    circleOfFriends: "",
    // 动态友链
    dynamicLink: {
      server: "",
      app_token: "",
      table_id: "",
    },
  },
  // 音乐播放器
  // https://github.com/imsyy/Meting-API
  music: {
    enable: false,
    // url：填自己的 Meting-API 地址（原来这里是上游留的占位符 api-meting.example.com）
    url: "",
    // id
    id: 9379831714,
    // netease / tencent / kugou
    server: "netease",
    // playlist / album / song
    type: "playlist",
  },
  // 搜索
  // https://www.algolia.com/
  search: {
    enable: false,
    appId: "",
    apiKey: "",
    // Algolia 的索引名（原来硬编码的是上游主题作者的 "imsyy"）
    indexName: "",
  },
  jumpRedirect: {
    enable: true,
    // 排除类名
    exclude: [
      "cf-friends-link",
      "upyun",
      "icp",
      "author",
      "rss",
      "cc",
      "power",
      "social-link",
      "link-text",
      "travellings",
      "post-link",
      "report",
      "more-link",
      "skills-item",
      "right-menu-link",
      "link-card",
    ],
  },
  // 站点统计
  tongji: {
    "51la": "",
  },
};
