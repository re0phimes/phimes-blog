import { tabsMarkdownPlugin } from "vitepress-plugin-tabs";
import markdownItAttrs from "markdown-it-attrs";
import container from "markdown-it-container";

// markdown-it
const markdownConfig = (md, themeConfig) => {
  // 插件
  md.use(markdownItAttrs);
  md.use(tabsMarkdownPlugin);

  // 修改默认的代码块渲染器来支持 Obsidian admonition
  const fence = md.renderer.rules.fence;
  md.renderer.rules.fence = (...args) => {
    const [tokens, idx] = args;
    const token = tokens[idx];
    const lang = token.info.trim();

    // 处理 Obsidian admonition
    if (lang.startsWith('ad-')) {
      const type = lang.substring(3); // 取ad-之后的内容，获取类型
      const content = token.content;

      const admonitionTypes = {
        'note': { title: 'NOTE', className: 'info' },
        'question': { title: 'QUESTION', className: 'info' },
        'warning': { title: 'WARNING', className: 'warning' },
        'tip': { title: 'TIP', className: 'tip' },
        'summary': { title: 'SUMMARY', className: 'summary' },
        'hint': { title: 'HINT', className: 'tip' },
        'important': { title: 'IMPORTANT', className: 'warning' },
        'caution': { title: 'CAUTION', className: 'warning' },
        'error': { title: 'ERROR', className: 'danger' },
        'danger': { title: 'DANGER', className: 'danger' }
      };

      const config = admonitionTypes[type] || { title: type.toUpperCase(), className: 'info' };

      return `<div class="${config.className} custom-block">
            <p class="custom-block-title" style="color: #fff; color: inherit;">${config.title}</p>
            ${md.render(content)}
    </div>`;
    }

    // 对于非 admonition 的代码块，使用原始的渲染器
    return fence(...args);
  };

  // timeline
  md.use(container, "timeline", {
    validate: (params) => params.trim().match(/^timeline\s+(.*)$/),
    render: (tokens, idx) => {
      const m = tokens[idx].info.trim().match(/^timeline\s+(.*)$/);
      if (tokens[idx].nesting === 1) {
        return `<div class="timeline">
                    <span class="timeline-title">${md.utils.escapeHtml(m[1])}</span>
                    <div class="timeline-content">`;
      } else {
        return "</div></div>\n";
      }
    },
  });
  // radio
  md.use(container, "radio", {
    render: (tokens, idx, _options, env) => {
      const token = tokens[idx];
      const check = token.info.trim().slice("radio".length).trim();
      if (token.nesting === 1) {
        const isChecked = md.renderInline(check, {
          references: env.references,
        });
        return `<div class="radio">
          <div class="radio-point ${isChecked}" />`;
      } else {
        return "</div>";
      }
    },
  });
  // button
  md.use(container, "button", {
    render: (tokens, idx, _options) => {
      const token = tokens[idx];
      const check = token.info.trim().slice("button".length).trim();
      if (token.nesting === 1) {
        return `<button class="button ${check}">`;
      } else {
        return "</button>";
      }
    },
  });
  // card
  md.use(container, "card", {
    render: (tokens, idx, _options) => {
      const token = tokens[idx];
      if (token.nesting === 1) {
        return `<div class="card">`;
      } else {
        return "</div>";
      }
    },
  });
  // 表格
  md.renderer.rules.table_open = () => {
    return '<div class="table-container"><table>';
  };
  md.renderer.rules.table_close = () => {
    return "</table></div>";
  };
  // 图片
  md.renderer.rules.image = (tokens, idx) => {
    const token = tokens[idx];
    const src = token.attrs[token.attrIndex("src")][1];
    const alt = token.content;
    if (!themeConfig.fancybox.enable) {
      return `<img src="${src}" alt="${alt}" loading="lazy">`;
    }
    return `<a class="img-fancybox" href="${src}" data-fancybox="gallery" data-caption="${alt}">
                <img class="post-img" src="${src}" alt="${alt}" loading="lazy" />
                <span class="post-img-tip">${alt}</span>
              </a>`;
  };

  // obsidian admonition containers
  // const admonitionTypes = {
  //   'note': { title: 'NOTE', className: 'info' },
  //   'question': { title: 'QUESTION', className: 'info' },
  //   'warning': { title: 'WARNING', className: 'warning' },
  //   'tip': { title: 'TIP', className: 'tip' },
  //   'summary': { title: 'SUMMARY', className: 'summary' }
  // };

  // Object.entries(admonitionTypes).forEach(([type, config]) => {
  //   md.use(container, `ad-${type}`, {
  //     validate: (params) => params.trim().match(new RegExp(`^ad-${type}`)),
  //     render: (tokens, idx, options, env) => {
  //       if (tokens[idx].nesting === 1) {
  //         // 开始标签
  //         return `<div class="custom-container ${config.className}">\n<p class="custom-container-title">${config.title}</p>\n`;
  //       } else if (tokens[idx].nesting === 0) {
  //         // 处理容器内容
  //         const content = tokens[idx].content;
  //         return md.render(content);
  //       } else {
  //         // 结束标签
  //         return '</div>\n';
  //       }
  //     },
  //     marker: '`'
  //   });
  // });
};

export default markdownConfig;
