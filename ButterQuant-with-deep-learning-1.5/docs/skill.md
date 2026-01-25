### 这个帖子在讲什么？
这个帖子是由用户 @LufzzLiz（岚叔）发布的，链接到一个X文章（ID: 2010299411411480576），主题是关于 **Claude AI 的 Skills 系统本地管理和分享**。从帖子内容、回复和相关搜索来看，它主要分享了一个实用教程或指南，焦点是如何使用 **OpenSkills** 这个开源库/CLI 工具来管理 Claude 的 Skills（技能）。

- **核心内容总结**：
  - Claude Skills 是 Anthropic 开发的 AI 功能：每个 Skill 是一个 Markdown 文件（SKILL.md），包含 YAML 元数据（如 name、description）和详细指令。Claude 会根据用户查询自动匹配并应用这些 Skills，实现特定任务如代码审查、生成 commit 消息、处理 PDF 等。
  - 帖子强调 **OpenSkills** 的作用：这是一个 CLI 工具（命令行界面），让 Claude Skills 变得“通用化”。它允许在本地安装、管理、同步和分享 Skills，甚至跨平台使用（不限于 Claude Code，还支持其他 AI IDE 如 Cursor、Aider、Windsurf）。
  - 为什么“可怕”或热门？OpenSkills 支持 **progressive disclosure**（渐进加载）：AI 只在需要时加载 Skill，避免提示过长导致幻觉。同时，它兼容 Claude 的热重载，能让 Skills 在不同工具间无缝共享。
  - 帖子回复中：
    - 有人确认用 OpenSkills 进行本地管理和分发，岚叔回复“是的，这个库我验过了OK的，推荐用这个管理skill”。
    - 讨论了 OpenSkills 的开发者（外国人，GitHub: numman-ali/openskills）。
    - 还有人提到 Antigravity 支持 Claude 模型（如 Opus 4.5），岚叔说“很早就支持了”并附图。
    - 其他回复如请求教程“如何创建自己的skill”，或抱怨想“一键同步所有skills”。

总体，这是 AI 开发者社区的分享帖，针对 Claude 用户，推荐 OpenSkills 来简化 Skills 管理。日期是 2026 年 1 月 11 日，热度高（5.8 万浏览，360 收藏）。

### 你可以用什么方法在 Antigravity 上应用？
好消息：**OpenSkills 完全可以应用到 Antigravity 上**！Antigravity（Google 的 AI-first IDE，基于 Gemini）没有原生 Claude Skills 支持，但 OpenSkills 的设计就是为了跨平台兼容。它通过生成一个 **AGENTS.md** 文件（或类似 rules 文件）来实现 Skills 的渐进加载，这和你的习惯（子文件夹放 .md，让 AI 读）高度契合。

- **为什么适合 Antigravity**？
  - Antigravity 强于项目上下文感知和 Rules 系统（.agent/rules.md 或 GEMINI.md），它能读整个 workspace 的文件，包括 AGENTS.md。
  - OpenSkills 用 CLI 方式同步 Skills 到 AGENTS.md，这个文件可以作为 Antigravity 的“全局指令库”，让 Gemini agent 像 Claude 一样渐进调用 Skills。
  - 从社区反馈，类似 Cursor（另一个 IDE）就用 OpenSkills + AGENTS.md 实现了 Claude Skills 的效果。Antigravity 的 agent 模式（多代理协作）也能类似处理。
  - 帖子中提到 Antigravity 支持 Claude 模型（通过 API 或代理），所以你可以混合使用：Gemini 做前端/规划，Claude 做复杂 Skills。

- **怎么用？一步步指南**
  假设你有 Node.js 和 npm（Antigravity 用户通常有，因为它支持 JS/TS 项目）。如果没有，先安装 Node.js。

  1. **安装 OpenSkills**（全局安装 CLI）：
     - 在终端运行：
       ```
       npm i -g openskills
       ```
     - 这会安装 OpenSkills 工具（GitHub: numman-ali/openskills）。它是免费开源的，兼容 Mac/Windows/Linux。

  2. **安装一些 Skills**（从市场或 GitHub）：
     - OpenSkills 支持从 Anthropic 的 Skills 市场安装（互动选择，默认放项目目录）：
       ```
       openskills install anthropics/skills
       ```
     - 或安装特定 Skills（如 PDF 处理、代码审查）：
       ```
       openskills install <github-repo-or-name>  # 例如 openskills install voltagent/awesome-claude-skills
       ```
     - Skills 会下载到 `~/.claude/skills/` 或项目 `.claude/skills/` 目录，每个 Skill 是一个子文件夹 + SKILL.md 文件。这和你的子文件夹习惯一样！

  3. **同步 Skills 到 AGENTS.md**（关键步，让 Antigravity 能用）：
     - 运行：
       ```
       openskills sync
       ```
     - 这会扫描你的 Skills 目录，生成一个 **AGENTS.md** 文件（放在项目根目录或 ~/.agent/）。文件内容是 XML 格式的 <available_skills> 块，列出所有 Skills 的 name 和 description。
     - 示例 AGENTS.md 片段：
       ```
       <available_skills>
         <skill name="pdf-extractor" description="Extract text and data from PDF files using advanced parsing techniques." />
         <skill name="code-review" description="Perform thorough code reviews following team standards." />
       </available_skills>
       ```
     - OpenSkills 支持 progressive disclosure：Antigravity 的 agent 看到列表后，只在匹配时用 `openskills read <skill-name>` 加载完整 SKILL.md（你可以用 prompt 教 agent 这么做）。

  4. **在 Antigravity 中集成和使用**：
     - 把 AGENTS.md 放进项目根目录或 .agent/ 子文件夹（Antigravity 会自动扫描）。
     - 如果你习惯用 rules.md，运行一个初始化 prompt 合并它们：
       - 在 Antigravity prompt 里输入：
         ```
         Scan AGENTS.md and incorporate all available skills as persistent guidelines. When a task matches a skill description, use `openskills read <skill-name>` to load it. Summarize and append to .agent/rules.md for future use.
         ```
       - 这会让你的 rules.md 包含 Skills 列表，形成“热重载”效果（改 Skills 后，再 sync 一次）。
     - 示例使用：
       - 想生成代码？Prompt："使用 code-review skill 帮我审查这个 React 组件。" Antigravity 的 agent 会匹配 description，加载 Skill，然后应用。
       - 如果 Antigravity 支持 Bash 工具（很多 AI IDE 支持），agent 可以直接调用 openskills CLI。
     - 混合 Claude：如果 Antigravity 已支持 Claude Opus 4.5（如帖子回复所说），在设置里切换模型，或用 proxy（如 OpenRouter）调用 Claude 来运行 Skills。

- **注意事项和优化**：
  - **兼容性**：Antigravity 以 Gemini 为主，如果 Skills 需要 Claude 特定功能，用代理切换模型。社区有模板如 antigravity-claude-proxy。
  - **你的习惯适配**：继续用子文件夹放 .md（现在是 Skills 文件夹），OpenSkills 帮你自动化汇总到 AGENTS.md，避免手动复制。
  - **测试**：先试一个小 Skill（如 Anthropic 的默认 pdf skill），sync 后在 Antigravity prompt 测试匹配。
  - **潜在问题**：如果 Antigravity 不直接支持 CLI 调用，纯用 AGENTS.md 作为 prompt 扩展也行（效果 80% 类似 Claude）。
  - **社区资源**：GitHub 上有 awesome-claude-skills 仓库，收集上百 Skills；Reddit r/ClaudeAI 有更多讨论。

这个方法能让你在 Antigravity 上模拟 Claude 的“可怕” Skills 热重载和自我进化，超级实用！如果安装卡壳或想具体 Skills 示例，告诉我，我帮你细化～🚀