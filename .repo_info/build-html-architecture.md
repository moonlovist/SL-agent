# HTML 文档架构与样式规范

> 适用于 `.repo_info/` 下所有文档页面。公共样式集中在 `common.css`，各页面通过 `<link>` 引用，仅保留页面专属覆盖和内联 JS。

---

## 1. 页面骨架

每个页面必须包含以下元素（按出现顺序）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title — LLM-Spectro-Agent</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../common.css">
</head>
<body>
  <!-- 侧边导航（必须） -->
  <nav class="sidebar" id="sidebar">...</nav>
  <button class="sidebar-toggle" id="sidebar-toggle" aria-label="Toggle navigation" aria-expanded="false">☰</button>
  <div class="sidebar-overlay" id="sidebar-overlay"></div>
  <!-- 光谱装饰条（必须） -->
  <div class="spectral-edge" aria-hidden="true"></div>
  <!-- 主内容区 -->
  <div class="container">
    <!-- 面包屑 / 标题+概览 / 函数树 / 目录 / 详细内容 / 注释 / 前后导航 / 页脚 -->
  </div>
  <script src="../common.js"></script>
  <script>/* 侧边栏 JS（见第 4 节） */</script>
</body>
</html>
```

**字体**：Google Fonts — `EB Garamond`（展示）、`IBM Plex Sans`（正文）、`IBM Plex Mono`（代码/标签）。子页面路径用 `../common.css`，index.html 用 `common.css`。

---

## 2. 公共组件

以下组件由 `common.css` 提供，页面内直接用 HTML 类名即可，无需写 CSS。

### 2.1 全局元素

| 组件 | HTML 模式 |
|---|---|
| 光谱装饰条 | `<div class="spectral-edge" aria-hidden="true"></div>` |
| 面包屑 | `<div class="breadcrumb"><a href="..">Home</a> / ...</div>` |
| 页脚 | `<div class="footer"><p><a href="..">Home</a> &middot; LLM-Spectro-Agent &middot; 2026-06-11</p></div>` |

### 2.2 内容组件

| 组件 | HTML 模式 | 用途 |
|---|---|---|
| 概览框 | `<div class="overview-box">...</div>` | 页面头部信息（路径、依赖、标签） |
| 标签行 | `<div class="tag-row"><span class="tag primary">...</span></div>` | 关键词标签，颜色见第 7 节 |
| 函数树 | `<div class="file-tree"><ul class="tree">...</ul></div>` | 函数/类结构树，支持 `<details>` 折叠 |
| 函数块 | `<div class="func-block" id="..."><h3>name</h3><div class="sig"><code>...</code></div><p>...</p></div>` | 单个函数/方法的详细说明 |
| 类标题 | `<h2 id="ClassName"><span class="class-badge">class</span> ClassName</h2>` | 类声明，后跟 `<p class="desc">` |
| 目录 | `<ol class="toc"><li><a href="#...">...</a></li></ol>` | 页内锚点目录 |
| 表格 | `<table><tr><th>Key</th><th>Type</th><th>Env Var</th><th>Description</th></tr>...</table>` | 字段/参数说明（四列标准格式） |
| 注释框 | `<div class="note"><strong>标题</strong> 内容</div>` | 重要提示/设计说明 |
| 前后导航 | `<div class="prev-next"><a href="...">← Prev</a> · <a href="...">Next →</a></div>` | 上一篇/下一篇链接 |

### 2.3 index.html 专属组件

以下组件仅在主页使用，CSS 同样来自 `common.css`：

| 组件 | HTML 模式 |
|---|---|
| 卡片网格 | `<div class="card-grid"><div class="card"><span class="badge done">Complete</span><h4><a>...</a></h4><div class="path">...</div><div class="desc">...</div><div class="deps">...</div></div></div>` |
| 状态徽章 | `<span class="badge done">Complete</span>` 或 `<span class="badge skel">Skeleton</span>` |
| 统计卡片 | `<div class="stat-grid"><div class="stat-card"><div class="num">5</div><div class="label">Stages</div></div></div>` |
| 分区标题 | `<div class="section-header"><h2>...</h2><span class="count">N modules</span></div><p class="section-intro">...</p>` |
| 英雄统计 | `<div class="hero-stats"><div class="hero-stat"><span class="num">5</span><span class="unit">Stages</span></div></div>` |
| 代码图表 | `<div class="graph">ASCII diagram</div>` |
| 标题副标签 | `<h1><span class="section-label">Core</span>Title</h1>` 或 `<h1><span class="sub">Label</span>FORMA</h1>` |

---

## 3. 侧边导航栏

### 3.1 结构模式

**index.html（顶级页面）**：扁平列表。

```html
<nav class="sidebar" id="sidebar">
  <a href="#" class="sidebar-link" data-label="Page top" id="sidebar-link-top">TOP</a>
  <a href="#pipeline" class="sidebar-link" data-label="Pipeline">Pipeline</a>
  <!-- 更多一级链接 ... -->
</nav>
```

**子页面（含可展开 Contents）**：

```html
<nav class="sidebar" id="sidebar">
  <a href="#" class="sidebar-link" data-label="Page top" id="sidebar-link-top">TOP</a>
  <a href="#function-tree" class="sidebar-link" data-label="Function tree">Tree</a>
  <div class="sidebar-section" id="sidebar-section-contents">
    <a href="#contents" class="sidebar-link sidebar-section-toggle" data-label="Contents">Contents</a>
    <div class="sidebar-sublist" id="sidebar-sublist-contents">
      <a href="#section-a" class="sidebar-link sidebar-sublink" data-label="section_a()">section_a</a>
      <a href="#section-b" class="sidebar-link sidebar-sublink" data-label="section_b()">section_b</a>
    </div>
  </div>
</nav>
```

**规则**：
- `id="sidebar-link-top"` 必须存在，作为回退高亮
- `id="sidebar-sublist-contents"` 和 `id="sidebar-section-contents"` 必须存在，JS 用其控制折叠
- 一级标签用全称，无标记符号。二级标签用全称，前缀实心圆点（CSS `::before` 自动）
- `data-label` 存完整名称，供移动端展开时替换文本
- `#function-tree` 和 `#contents` 对应页内 h2 的 id

### 3.2 侧边栏 JS 模板

每页必须内联 JS（在 `</body>` 前）。自定义：

1. **`contentsSubIds`** — 属于 Contents 区块的所有锚点 id（含 `'contents'` 自身）
2. 无二级导航的页面（index.html, config_overview）省略 `sublistContents`/`contentsSubIds`

参考 `llm.html`、`main.html` 或任一 `config_*.html` 获取完整模板。

核心行为：
- scroll-spy：从下往上找 `top ≤ 120px` → 高亮；触底时放宽为 `top < 屏高`
- 移动端 ☰ 展开全高侧边栏，点击遮罩/链接/Escape 关闭
- Contents 点击立即展开子列表
- 所有非锚点链接自动 `target="_blank"`

---

## 4. index.html 专属样式

`index.html` 在 `<style>` 块中覆盖：

- 正文字号 18px，容器 max-width 1100px
- h1/h2/h3 更大的展示字号，h2::before 用 emission-dim 竖线
- Pipeline 布局全系列：`.pipeline`、`.pipeline-stage`、`.pipeline-stage-inner`、`.pipeline-arena`、`.arena-ha`、`.arena-aa-col`、`.sub-block`、`.connector-*`、`.two-col`
- Module 标签（`.module-label`）：arena-head 内绝对定位在左侧
- 响应式 768px 折叠

子页面通常无需额外 `<style>` 块。

---

## 5. 内容核对清单

- [ ] 路径匹配源代码（`src/FORMA/` 非旧的 `src/AstroAgent/`）
- [ ] 函数签名、类字段、默认值、env var 名与源码一致
- [ ] 不存在的字段/方法已移除
- [ ] 外部链接用相对路径，JS 自动 `target="_blank"`
- [ ] 页内锚点 `#id` 与目标 `id` 一致
- [ ] `Function Tree`、`Contents` h2 有 `id="function-tree"` 和 `id="contents"`

---

## 5b. 函数与类的 Input/Output 表格规范

每个 `func-block` 必须配有结构化的 Input/Output 表格，取代纯文本描述参数和返回值。具体 header 模式、`<colgroup>` 模板和适用场景见**第 8 节**。

### 5b.1 func-block 内元素顺序

1. `<h3>` 函数名
2. `<div class="sig"><code>…</code></div>` 签名
3. `<p>` 一句话概述
4. `<table class="table-params">` 参数表（有参数时写入）
5. `<table class="table-output">` 返回值表（有返回值时写入；`-> None` 则省略）

嵌套在 func-block 内的表格会自然继承 func-block 的内边距和背景，无需额外包裹。

### 5b.2 类字段 / 配置项

类级别的字段、配置项等结构化键值文档使用 `table-keys`，按实际列数选择模板（见第 8.2 节）。

---

## 6. 主题

色彩变量定义在 `common.css` 的 `:root`（暗色）和 `[data-theme="light"]`（亮色）。页面内用 `var(--xxx)` 引用，不硬编码色值。

**暗色**：deep blue-black 背景、amber 标题 (`#eccd70`)、emission 橙强调 (`#ff8868`)、absorption 蓝链接 (`#6ebeff`)。

**亮色**：浅灰背景、crimson 标题 (`#9c0c13`)、蓝色强调和链接。

主题切换按钮由 `common.js` 提供（右下角 ☀/☽），localStorage 记忆，首次跟随 OS 偏好。

---

## 7. Tag 颜色速查

| class | 暗色 | 亮色 |
|---|---|---|
| `tag primary` | 暖橙底 + emission 橙字 | 深蓝底 + emission 蓝字 |
| `tag secondary` | 暖琥珀底 + amber-dim 金字 | 淡蓝底 + absorption 蓝字 |
| `tag muted` | 极淡金底 + faint 灰字 | 极淡灰底 + faint 灰字 |
| `tag green` | green 底 + green 字 | green 底 + green 字 |

---

## 8. 表格列宽比例

三类表格通过 `<colgroup>` + CSS class 控制列宽。最后一栏（Description）不设宽度，自动填充剩余空间。

以下为所有合法 header 模式。**新增页面的表格 header 必须从这些模式中选择，不得自创变体。**

### 8.1 CSS 预设（定义在 `common.css`）

| class | 宽度 | 用途 |
|---|---|---|
| `.col-name` | 20% | Key / Field / Parameter / Method 名称 |
| `.col-type` | 14% | Type 类型标注 |
| `.col-env` | 22% | Env Var 环境变量名 |
| `.col-def` | 12% | Default 默认值 |
| `.col-source` | 18% | Source 来源（仅 table-params） |
| `.col-out` | 26% | Output 输出类型（仅 table-output，对齐 params 的 Default 位置） |

### 8.2 table-keys（蓝 header）— 字段 / 配置项

用于类字段、配置项、State 字段等结构化键值文档。

| # | 合法 header | `<colgroup>` | 适用场景 |
|---|---|---|---|
| K3 | `Key \| Type \| Description` | `<col class="col-name"><col class="col-type"><col>` | 简单字段表（State 字段等） |
| K4a | `Key \| Type \| Env Var \| Description` | `<col class="col-name"><col class="col-type"><col class="col-env"><col>` | 配置项（有环境变量映射） |
| K4b | `Field \| Type \| Default \| Description` | `<col class="col-name"><col class="col-type"><col class="col-def"><col>` | 类变量（有默认值，无环境变量） |
| K5 | `Field \| Type \| Env Var \| Default \| Description` | `<col class="col-name"><col class="col-type"><col class="col-env"><col class="col-def"><col>` | 完整配置项（环境变量 + 默认值） |

> **规则**：K4a/K5 的第一栏也可写 `Field`（类变量语境）或 `Key`（配置语境），含义相同。**不得**出现 `Key \| Type \| Default \| Description` 这种缺少 Env Var 的 4 栏模式——那属于 K4b。

### 8.3 table-params（琥珀 header）— 函数参数

统一 4 栏。用于 func-block 内的函数/方法参数文档。

| # | 合法 header | `<colgroup>` | 适用场景 |
|---|---|---|---|
| P4a | `Parameter \| Type \| Default \| Description` | `<col class="col-name"><col class="col-type"><col class="col-def"><col>` | 普通函数参数，有默认值写值，无则填 `—` |
| P4b | `Parameter \| Type \| Source \| Description` | `<col class="col-name"><col class="col-type"><col class="col-source"><col>` | 构造函数参数，值来自外部注入（如 DI 容器） |

> **规则**：不再使用 3 栏模式。所有 params 表统一 4 栏。P4a 是默认选择；P4b 仅用于构造注入场景，Source 栏填写注入来源（如 `AllConfig`、`Injected`）。

### 8.4 table-output（绿 header）— 返回值

用于 func-block 内的函数返回值文档。无返回值的函数不写此表。

| # | 合法 header | `<colgroup>` | 适用场景 |
|---|---|---|---|
| O2 | `Method \| Description` | `<col class="col-name"><col>` | 仅列方法名和说明（内部 helper 列表等） |
| O3a | `Output \| Type \| Description` | `<col class="col-name"><col class="col-out"><col>` | 单个返回值（绝大多数函数的默认选择） |
| O3b | `Method \| Output \| Description` | `<col class="col-name"><col class="col-out"><col>` | 方法列表 + 输出格式（如 ResultWriter 的 write 方法一览） |

### 8.5 快速对照

```
table-keys  ──  K3  K4a  K4b  K5   （蓝 header）
table-params──            P4a  P4b  （琥珀 header，统一 4 栏）
table-output──  O2  O3a  O3b        （绿 header）
```
