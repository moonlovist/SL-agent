from pathlib import Path
from html import escape
import json

OUT = Path(__file__).resolve().parent

def diagram(name, title, nodes, edges, width, height):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#64748b"/></marker></defs>', '<rect width="100%" height="100%" fill="#f8fafc"/>', f'<text x="32" y="40" font-family="sans-serif" font-size="22" font-weight="bold" fill="#0f172a">{escape(title)}</text>']
    for a,b,label in edges:
        x,y,w,h,*_=nodes[a]; X,Y,W,H,*_=nodes[b]
        if Y>=y+h:
            sx,sy,tx,ty=x+w/2,y+h,X+W/2,Y
            path=f'M{sx},{sy} V{(sy+ty)/2} H{tx} V{ty}'
        else:
            sx,sy,tx,ty=x+w,y+h/2,X,Y+H/2
            path=f'M{sx},{sy} H{(sx+tx)/2} V{ty} H{tx}'
        parts.append(f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
        if label:
            parts.append(f'<text x="{(sx+tx)/2+6}" y="{(sy+ty)/2-6}" font-family="sans-serif" font-size="13" fill="#475569">{escape(label)}</text>')
    for x,y,w,h,lines,color in nodes.values():
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#cbd5e1"/>')
        for i,line in enumerate(lines):
            parts.append(f'<text x="{x+w/2}" y="{y+27+i*23}" text-anchor="middle" font-family="sans-serif" font-size="{16 if i==0 else 14}" fill="#0f172a">{escape(line)}</text>')
    parts.append('</svg>')
    (OUT/name).write_text(''.join(parts),encoding='utf-8')

blue='#dbeafe'; green='#dcfce7'; yellow='#fef3c7'; grey='#e2e8f0'
nodes={}
stages=[('FITS 科学数据','Flux · RMS · PSF · 多波段'),('预处理与前景中心','人类修改中心或掩膜后完整重跑'),('前景光模型拟合与扣除','残差图与另一前景模型的比较'),('候选搜索与数值测量','多阈值 · SEP · 几何路线 · 形态 · 颜色'),('统一证据准备','真实像素合并 · 去重 · 重算场景几何'),('规则专家判断 → 规则审计','多假设比较；检查引用与判断一致性'),('可选 DeepSeek 两阶段评审','接收匿名化测量；不接收规则等级'),('报告与 UI','规则分类负责分组；模型评审单独显示'),('事后评估','此时才加入官方人类标签进行比较')]
for i,lines in enumerate(stages): nodes[str(i)]=(95,75+i*106,650,76,lines,blue if i<5 else green if i<7 else grey)
diagram('01_overall.svg','当前实际执行流程',nodes,[(str(i),str(i+1),'') for i in range(8)],840,1040)

nodes={
'q':(220,75,520,76,['数据与噪声信息可用吗','否 → 证据不足并弃权'],grey),
'c':(220,200,520,76,['是否形成清晰透镜构型','空间关系与形态共同支持'],blue),
'conf':(70,330,410,100,['清晰构型：检查关键冲突','强替代解释或模型符号冲突 → C','可靠颜色冲突或稳健性不足 → B'],yellow),
'weak':(530,330,410,100,['构型不清晰：检查特异性','可信且具透镜结构关系 → B','不完整线索 → C；缺乏支持 → X'],yellow),
'a':(70,495,410,100,['A 类规则条件','清晰构型 + 另一前景模型支持','无上述关键冲突'],green),
'audit':(220,675,520,100,['规则审计与发布','审计失败不能正常发布 A/B/C','影子几何建议不会改写正式规则类别'],grey)}
diagram('02_rule_grading.svg','规则分类逻辑概览',nodes,[('q','c','是'),('c','conf','是'),('c','weak','否'),('conf','a','无冲突且有支持'),('a','audit',''),('weak','audit','')],1010,825)

nodes={}
stages=[('匿名化规范化证据 + 领域知识','去除身份、路径、官方标签与规则等级'),('第一阶段：逐结构解释与假设生成','必须覆盖全部结构；同时提出透镜与非透镜假设'),('程序验证证据使用','字段存在 · 数值一致 · 结构覆盖 · 格式合法'),('第二阶段：假设裁决','比较支持与反证，提出类别、限制和待测信息'),('再次验证并保存评审','两阶段共享最多一次修复机会'),('与规则结果并列展示','API 或验证失败：记录失败，保留规则结果')]
for i,lines in enumerate(stages): nodes[str(i)]=(60,80+i*115,740,80,lines,blue if i in [0,2,4] else green)
diagram('03_deepseek.svg','DeepSeek 评审与程序验证',nodes,[(str(i),str(i+1),'') for i in range(5)],860,790)

body='''
<header><p class="eyebrow">SL agent · 架构说明 · 2026 年 9 月 30 日</p><h1>强引力透镜智能体逻辑分析图谱</h1><p class="lead">系统先把图像转换成可追溯的测量证据，再进行规则判断与可选大模型评审，最终生成报告并与人类标签比较。当前大模型是实验性评审层，尚未取代规则分类，也尚未完成透镜物理闭合验证。</p><p>代码基线：feature/deepseek-judgment 分支，43347d9。本文描述现有实现；最后一节单独说明建议扩展。</p></header>
<nav><a href="#overall">整体流程</a><a href="#evidence">证据形成</a><a href="#rules">分类逻辑</a><a href="#llm">模型评审</a><a href="#output">输出与评估</a><a href="#future">后续闭环</a></nav>
<section id="overall"><h2>整体流程与职责边界</h2><img src="01_overall.svg" alt="当前实际执行流程图"><p>执行顺序是数值分析、统一证据准备、规则判断、规则审计、可选 DeepSeek、报告。两种判断共享规范化测量；DeepSeek 不接收规则等级。官方目录标签仅用于 UI 对照和事后评估。</p><p>候选检测负责找到信号，测量负责描述信号，分类负责解释信号。搜索路线的数量、候选碎片的数量都不能直接当成多个透镜像或独立证据。</p></section>
<section id="evidence"><h2>从图像形成可追溯证据</h2><table><tr><th>步骤</th><th>产生的信息</th><th>解释边界</th></tr><tr><td>数据读取</td><td>通量、噪声、PSF、多波段信息</td><td>PNG 使用降级模式，不具备 FITS 的校准信息</td></tr><tr><td>前景拟合</td><td>星系光模型、残差与模型比较</td><td>残差不自动等于背景源；另一模型仍来自同一观测</td></tr><tr><td>候选搜索</td><td>多阈值、SEP、几何路线发现的结构</td><td>几何路线是发现过程，不能单独证明透镜</td></tr><tr><td>结构测量</td><td>SNR、半径、切向关系、角度覆盖、径向宽度、颜色</td><td>颜色一致只提供支持，不能创造透镜构型</td></tr><tr><td>统一证据准备</td><td>可测量结构去重，弧段真实像素合并，重算场景几何</td><td>不填补弧段间隙，不创造缺失信号</td></tr></table><p>当前默认测量准入条件包含正通量、有限数值、至少 5 个像素、积分 SNR ≥ 4。这是进入证据集合的门槛，不是 A/B 分类门槛。原始 Observation 保留检测过程；规范化 Observation 为判断提供一致的结构集合。</p><p>人类修改前景中心或掩膜会触发完整重跑，因为前景扣除的变化会影响候选、测量和最终类别。</p></section>
<section id="rules"><h2>规则如何区分 A B C X</h2><img src="02_rule_grading.svg" alt="规则分类概览"><table><tr><th>类别</th><th>当前规则含义</th></tr><tr><td>A</td><td>清晰透镜构型，有另一前景模型的信号支持，没有关键模型冲突、可靠颜色冲突或同样强的替代解释。</td></tr><tr><td>B</td><td>可信结构具有透镜特异性但构型不够清晰；或形态清晰而稳健性、颜色等仍有限制。</td></tr><tr><td>C</td><td>存在可能的透镜线索，但构型、测量或竞争解释仍未解决。</td></tr><tr><td>X</td><td>数据可用，但当前证据缺乏可信的透镜特异性结构关系。</td></tr><tr><td>证据不足</td><td>数据质量或必要噪声信息不满足判断条件，弃权而不是归入 X。</td></tr></table><p>分类不是各项分数相加，也没有输出经过校准的透镜概率。缺少测量与测得反证分开处理；前景颜色相似本身不会自动限制 A，可靠的候选像间颜色冲突才可能限制 A。</p><p>代码保留了“同样强的非透镜解释”等分支，但部分起源判别尚未实现专门测试。例如亮度峰不等于已确认星系核，残差不稳定也不等于已确认 PSF 伪影。规则中另有更严格的 A 类几何影子检查，目前仅记录复核建议，不改写正式类别。</p></section>
<section id="llm"><h2>DeepSeek 如何参与判断</h2><img src="03_deepseek.svg" alt="DeepSeek 两阶段评审图"><p>模型接收结构化数值证据与透镜几何、弧形态、多波段、星系本征形态、扣除伪影和相互作用等领域知识；当前接口不是直接读取原始图像的多模态专家。</p><p>第一阶段要求逐结构解释并提出透镜、非透镜假设；程序验证后，第二阶段比较假设并输出分类、理由、限制和缺失信息。两个阶段使用同一模型，不能解释成两个独立专家投票。</p><table><tr><th>已经验证</th><th>尚未因此验证</th></tr><tr><td>引用字段存在且有值；数值一致；结构覆盖完整；输出格式及类别处置合法</td><td>测量本身无系统误差；解释符合透镜方程；多个像来自同一背景源；A/B 与人类一致</td></tr></table><p>两阶段共享最多一次修复机会。调用或验证失败时保存失败状态，保留规则结果，不产生伪有效模型分类。此流程借鉴 FORMA 的证据、假设、可执行检查和裁决组织方式；其光谱任务效果不能直接推广到强透镜分类。</p><p>参考：<a href="https://arxiv.org/abs/2607.06128">Executable verification through formalized expert reasoning in astronomical spectroscopy</a>。</p></section>
<section id="output"><h2>输出 评估与当前结果</h2><p>规则分类负责 UI 分组与发布字段；规则审计失败时不能正常发布 A/B/C。模型分类单独展示为实验性结果，不与规则平均，也不自动覆盖规则。审计检查引用和规则一致性，不是独立科学专家复核。</p><p>每个对象保存原始与规范化 Observation、reasoning.json、summary.json、report.md，以及可选 llm_input.json、llm_assessment.json、代码快照和运行清单。模型输入不含官方标签、对象路径或规则等级；运行记录不保存 API 密钥。</p><div class="callout">当前 10 个官方 A/B 样本中，规则等级一致率为 4/10。最新版 DeepSeek 批量调用均因 HTTP 402 余额不足失败，因此没有这一轮有效模型分类结果。这不代表模型准确率为零。样本规模小且缺少 C/X 对照，不能用于证明人类水平或评估污染率。</div><p>这里的“一致率”比较 A/B 等级是否相同，不等于透镜检出率。评估分别记录可用结果覆盖率、全部目标一致情况及有效分类中的一致情况，避免把 API 失败当作科学判断。</p></section>
<section><h2>代码对应关系</h2><table><tr><th>责任</th><th>主要文件 位于 SL-agent/src/expert_agent</th></tr><tr><td>数值分析</td><td>vision/pipeline.py</td></tr><tr><td>统一证据</td><td>agents/evidence_preparation.py</td></tr><tr><td>规则判断与审计</td><td>agents/expert_assessment_agent.py · agents/decision_audit.py</td></tr><tr><td>工作流顺序</td><td>core/workflow.py</td></tr><tr><td>大模型协议与调用</td><td>agents/llm_assessment.py · core/llm_client.py · schemas/llm_reasoning.py</td></tr><tr><td>证据引用验证</td><td>tools/evidence_verification.py</td></tr><tr><td>报告与复现</td><td>agents/report_writer.py · core/reproducibility.py</td></tr><tr><td>界面与评估</td><td>ui/app.py · ui/project.py · core/evaluation.py</td></tr></table><p>上述 expert_agent 是当前运行架构。仓库中历史 FORMA 代码不代表当前工作流另有一套正在运行的专家团队。</p></section>
<section id="future"><h2>后续需要补齐的验证闭环</h2><p>以下为建议扩展，尚未实现自动闭环：模型定位不确定性 → 选择针对性的测量或拟合工具 → 产生新证据 → 再次比较假设 → 更新分类。</p><table><tr><th>待补环节</th><th>要回答的问题</th></tr><tr><td>透镜质量模型与源重建</td><td>多个候选像能否映射到同一个背景源？</td></tr><tr><td>透镜方程与物理闭合</td><td>候选构型能否由一致的物理模型解释？</td></tr><tr><td>替代解释专门检验</td><td>盘、螺旋臂、相互作用或 PSF 伪影是否更合理？</td></tr><tr><td>人类标签校准及独立留出评估</td><td>分类边界是否稳定，并能推广到未见数据？</td></tr><tr><td>主动补测与停止条件</td><td>还需哪项证据，何时应继续验证或明确弃权？</td></tr></table><p>优先方向是让模型提出的“缺什么证据”变成可执行的工具任务，同时建立包含 A/B/C/X 的独立评估集。完成这些检验前，当前系统应称为可追溯的实验性判断框架。</p></section>
'''
css='''body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#1e293b;background:#f1f5f9;margin:0;line-height:1.8}main{max-width:1040px;margin:32px auto;background:white;padding:48px 56px;border-radius:16px}h1{font-size:34px;line-height:1.4}h2{font-size:24px;margin-top:44px;color:#0f172a}p{margin:16px 0}.eyebrow{color:#64748b;font-size:14px}.lead{font-size:20px}nav{display:flex;flex-wrap:wrap;gap:18px;border-block:1px solid #e2e8f0;padding:16px 0}a{color:#2563eb}img{display:block;width:100%;max-width:860px;margin:24px auto;border:1px solid #e2e8f0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;vertical-align:top;padding:12px;border:1px solid #e2e8f0}th{background:#eff6ff}.callout{background:#fffbeb;border-left:4px solid #f59e0b;padding:18px;margin:20px 0}@media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:28px}table{font-size:13px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none}nav{display:none}h2{break-after:avoid}img{max-height:230mm;object-fit:contain;break-inside:avoid}tr{break-inside:avoid}section{margin-bottom:24px}}'''
(OUT/'SL-agent逻辑分析图谱.html').write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>强引力透镜智能体逻辑分析图谱</title><style>{css}</style></head><body><main>{body}</main></body></html>',encoding='utf-8')
print('Generated HTML document and three SVG diagrams in', OUT)
