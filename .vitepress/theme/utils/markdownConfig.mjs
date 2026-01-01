import { tabsMarkdownPlugin } from "vitepress-plugin-tabs";
import markdownItAttrs from "markdown-it-attrs";
import container from "markdown-it-container";

// SVG 图标
const icons = {
  question: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M11 18h2v-2h-2v2zm1-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.21 0-4 1.79-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 2-3 1.75-3 5h2c0-2.25 3-2.5 3-5 0-2.21-1.79-4-4-4z"/></svg>',
  note: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
  warning: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 2L1 21h22L12 2zm0 4l7.5 13h-15L12 6zm-1 5v4h2v-4h-2zm0 6v2h2v-2h-2z"/></svg>',
  tip: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18h6m-5 4h4M12 2a7 7 0 0 0-4 12.7V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.3A7 7 0 0 0 12 2z"/></svg>',
  danger: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15" stroke="white" stroke-width="2"/><line x1="9" y1="9" x2="15" y2="15" stroke="white" stroke-width="2"/></svg>',
  info: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
};

// Obsidian callout 类型映射
const calloutTypes = {
  'note': { title: 'NOTE', className: 'tip', icon: icons.note },
  'question': { title: 'QUESTION', className: 'question', icon: icons.question },
  'warning': { title: 'WARNING', className: 'warning', icon: icons.warning },
  'tip': { title: 'TIP', className: 'tip', icon: icons.tip },
  'summary': { title: 'SUMMARY', className: 'tip', icon: icons.note },
  'hint': { title: 'HINT', className: 'tip', icon: icons.tip },
  'important': { title: 'IMPORTANT', className: 'warning', icon: icons.warning },
  'caution': { title: 'CAUTION', className: 'warning', icon: icons.warning },
  'error': { title: 'ERROR', className: 'danger', icon: icons.danger },
  'danger': { title: 'DANGER', className: 'danger', icon: icons.danger },
  'info': { title: 'INFO', className: 'info', icon: icons.info },
};

// markdown-it
const markdownConfig = (md, themeConfig) => {
  // 插件
  md.use(markdownItAttrs);
  md.use(tabsMarkdownPlugin);

  // 处理 Obsidian callout 语法: > [!type] title
  md.core.ruler.after('block', 'obsidian-callout', (state) => {
    const tokens = state.tokens;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].type !== 'blockquote_open') continue;

      // 找到 blockquote 内的第一个 inline
      let inlineIdx = -1;
      for (let j = i + 1; j < tokens.length && tokens[j].type !== 'blockquote_close'; j++) {
        if (tokens[j].type === 'inline') { inlineIdx = j; break; }
      }
      if (inlineIdx === -1) continue;

      const inline = tokens[inlineIdx];
      const match = inline.content.match(/^\[!(\w+)\]\s*(.*?)(?:\n|$)/i);
      if (!match) continue;

      const type = match[1].toLowerCase();
      const customTitle = match[2].trim();
      const config = calloutTypes[type] || { title: type.toUpperCase(), className: 'info' };
      const title = customTitle || config.title;

      // 替换 blockquote_open 为 html_block
      tokens[i].type = 'html_block';
      const icon = config.icon || '';
      tokens[i].content = `<div class="${config.className} custom-block"><p class="custom-block-title">${icon} ${title}</p>`;
      tokens[i].tag = '';

      // 移除第一行的 [!type] 标记
      inline.content = inline.content.replace(/^\[!\w+\]\s*[^\n]*\n?/, '');
      if (inline.children) {
        // 重建 children，移除 callout 标记
        const newChildren = [];
        let skipNext = false;
        for (const child of inline.children) {
          if (skipNext && child.type === 'softbreak') { skipNext = false; continue; }
          if (child.type === 'text' && child.content.match(/^\[!\w+\]/)) {
            child.content = '';
            skipNext = true;
          }
          newChildren.push(child);
        }
        inline.children = newChildren.filter(c => c.content !== '' || c.type !== 'text');
      }

      // 找到对应的 blockquote_close 并替换
      let depth = 1;
      for (let j = i + 1; j < tokens.length; j++) {
        if (tokens[j].type === 'blockquote_open') depth++;
        if (tokens[j].type === 'blockquote_close') {
          depth--;
          if (depth === 0) {
            tokens[j].type = 'html_block';
            tokens[j].content = '</div>';
            tokens[j].tag = '';
            break;
          }
        }
      }
    }
  });

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
            <p class="custom-block-title">${config.title}</p>
            <div class="custom-block-content">
              ${md.render(content)}
            </div>
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
