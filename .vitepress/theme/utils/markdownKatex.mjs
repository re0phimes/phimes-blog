import katex from "katex";

/**
 * 数学公式渲染（KaTeX）
 *
 * 直接使用项目依赖里的 katex，而不是 markdown-it-katex。
 * markdown-it-katex 内部锁定了 katex@0.6.0（2017 年），解析不了
 * \begin{align} / \text{中文} / \operatorname / \begin{pmatrix} 等写法，
 * 而且失败时会静默把裸 LaTeX 当正文输出，导致正文里出现大段公式源码。
 *
 * 语法支持：
 *   $...$      行内公式
 *   $$...$$    行内公式（作者常把它写在段落中间，也一并支持）
 *   $$ 独占行   块级公式
 */

const DOLLAR = 0x24; // "$"
const SPACE = 0x20;
const TAB = 0x09;

const KATEX_OPTIONS = {
  throwOnError: false,
  strict: "ignore",
  errorColor: "#c64545",
  trust: false,
};

/**
 * 判断 pos 位置的 $ 能否作为起始/结束定界符。
 * 规则参考 markdown-it-katex，避免把 "$5" 这类货币写法当成公式。
 */
const checkDelimiter = (state, pos) => {
  const prev = pos > 0 ? state.src.charCodeAt(pos - 1) : -1;
  const next = pos + 1 <= state.posMax ? state.src.charCodeAt(pos + 1) : -1;

  return {
    // 后面紧跟空白，不能作为起始
    canOpen: next !== SPACE && next !== TAB,
    // 前面是空白，或后面紧跟数字，不能作为结束
    canClose:
      prev !== SPACE && prev !== TAB && !(next >= 0x30 && next <= 0x39),
  };
};

/** 从 from 开始找下一个未被转义的定界符 */
const findClosing = (state, from, delimiter) => {
  let pos = from;
  while ((pos = state.src.indexOf(delimiter, pos)) !== -1) {
    let escape = pos - 1;
    while (state.src[escape] === "\\") escape -= 1;
    // 定界符前是偶数个反斜杠 → 真的是定界符
    if ((pos - escape) % 2 === 1) return pos;
    pos += delimiter.length;
  }
  return -1;
};

const mathInline = (state, silent) => {
  const start = state.pos;
  if (state.src.charCodeAt(start) !== DOLLAR) return false;

  const delimiter = state.src.startsWith("$$", start) ? "$$" : "$";
  if (!checkDelimiter(state, start).canOpen) return false;

  const contentStart = start + delimiter.length;
  const close = findClosing(state, contentStart, delimiter);
  if (close === -1 || close === contentStart) return false;
  if (!checkDelimiter(state, close).canClose) return false;

  const content = state.src.slice(contentStart, close);
  if (!content.trim()) return false;

  if (!silent) {
    const token = state.push("math_inline", "math", 0);
    token.markup = delimiter;
    token.content = content;
  }

  state.pos = close + delimiter.length;
  return true;
};

const mathBlock = (state, start, end, silent) => {
  const firstLineStart = state.bMarks[start] + state.tShift[start];
  const firstLineEnd = state.eMarks[start];
  if (firstLineStart + 2 > firstLineEnd) return false;
  if (state.src.slice(firstLineStart, firstLineStart + 2) !== "$$") return false;

  const firstLine = state.src.slice(firstLineStart + 2, firstLineEnd);

  // $$ ... $$ 写在同一行
  if (firstLine.trim().endsWith("$$")) {
    if (silent) return true;
    state.line = start + 1;
    const token = state.push("math_block", "math", 0);
    token.block = true;
    token.markup = "$$";
    token.content = firstLine.trim().slice(0, -2).trim();
    token.map = [start, state.line];
    return true;
  }

  // 多行公式：向下找到以 $$ 结尾的那一行
  let next = start;
  let lastLine = "";
  let found = false;
  for (next = start + 1; next < end; next += 1) {
    const lineStart = state.bMarks[next] + state.tShift[next];
    const lineEnd = state.eMarks[next];
    if (lineStart < lineEnd && state.tShift[next] < state.blkIndent) break;

    const line = state.src.slice(lineStart, lineEnd);
    if (line.trim().endsWith("$$")) {
      lastLine = line.slice(0, line.lastIndexOf("$$"));
      found = true;
      break;
    }
  }
  if (!found) return false;
  if (silent) return true;

  state.line = next + 1;
  const token = state.push("math_block", "math", 0);
  token.block = true;
  token.markup = "$$";
  token.content =
    (firstLine.trim() ? `${firstLine}\n` : "") +
    state.getLines(start + 1, next, state.tShift[start], true) +
    lastLine;
  token.map = [start, state.line];
  return true;
};

const renderMath = (latex, displayMode, md) => {
  try {
    return katex.renderToString(latex, { ...KATEX_OPTIONS, displayMode });
  } catch (error) {
    // 正常不会走到这里（throwOnError: false 已经在公式内部标红），
    // 兜底也不能把裸 LaTeX 直接当正文吐出去。
    const escaped = md.utils.escapeHtml(latex);
    return `<code class="math-error" title="${md.utils.escapeHtml(error.message)}">${escaped}</code>`;
  }
};

export const markdownKatex = (md) => {
  md.inline.ruler.after("escape", "math_inline", mathInline);
  md.block.ruler.after("blockquote", "math_block", mathBlock, {
    alt: ["paragraph", "reference", "blockquote", "list"],
  });

  md.renderer.rules.math_inline = (tokens, idx) =>
    renderMath(tokens[idx].content, false, md);
  md.renderer.rules.math_block = (tokens, idx) =>
    `${renderMath(tokens[idx].content, true, md)}\n`;
};

export default markdownKatex;
