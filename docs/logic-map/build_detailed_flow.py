"""Generate an inspectable SVG of the implemented workflow, without network calls."""
from pathlib import Path
from html import escape
import json
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
# Each column is a sequential expansion; side exits are explicit nodes.
groups=[
('01 入口与科学数据',[
('入口选择','CLI analyze：数值分析后判断','UI：项目索引 → 子进程运行 CLI','reason：读取已有 Observation，跳过图像分析'),
('人工前景复核入口','foreground-review：中心 / 拟合掩膜 / 排除掩膜','使用原生像素坐标；修改后完整重跑','不会仅修改报告中的等级'),
('配置装配','CLI backend/model 优先于环境配置','环境变量优先于当前目录 .env','库 ExpertWorkflow() 默认离线规则模式'),
('载入图像 load_image','FITS：默认 VIS；Flux / RMS / PSF','读取 WCS、像素尺度与有效像素比例','PNG：缺乏校准通量、科学 RMS 和完整 PSF'),
('PNG 降级分支','SNR 层改为 3 / 4 / 6；保留次数 ≥ 2','SEP 网格缩小为 nthresh=16 / cont=0.01','检测门槛提高；记录科学信息不足'),
('预处理 preprocess','背景与噪声估计；初始源检测和分割','选择目标前景中心并检查中心歧义','定位拟合区域、源掩膜和初始光模型'),
('应用人工覆盖（如提供）','apply_foreground_override','中心和掩膜参与后续前景模型拟合','人工排除区域进入限制与溯源记录'),
('建立模型拟合输入','源保护权重：避免吸收候选背景信号','前景几何来自选定星系的中心对称光','有效像素 / RMS / PSF 决定拟合可用性'),
('数据与标签分离','对象 ID 用于输出与索引，不用于等级','A/B 文件夹标签用于 UI 对照和评估','不把官方目录标签送入判断提示词'),
]),
('02 前景光模型与残差',[
('拟合前景模型库','固定几何 PSF 卷积 MGE','双几何 MGE；有界网格双 Sérsic','均为前景光模型，不是透镜质量模型'),
('模型选择与质量记录','评估模型得分及前景拟合质量','记录候选、选中方法、质量与失败标记','成功模型可成为主扣除模型'),
('主残差与比较残差','主残差 = 数据 − 选中前景光模型','保留另一前景扣除用于敏感性比较','两者来自同一曝光，不是独立观测'),
('主模型失败分支','不运行正式阈值 / SEP / 几何提取','相应分割、足迹和候选集合置空','仍保留质量信息，交由后续判断处理'),
('可选自适应残差噪声校准','要求模型成功且 adaptive_residual_snr 开启','控制区：r > 1.5 × ring_outer，排除掩膜','至少 100 个有限残差 SNR 像素'),
('计算有效检测门槛','控制区中位数 + robust scale × 基础阈值','scale 限制在 0.60～2.50；门槛下限 1','同步更新阈值层、SEP 和结构低门槛'),
('控制区不足分支','不套用自适应变换，使用原配置','记录实际运行参数和校准说明','检测门槛不是 A/B 科学分类边界'),
('另一模型的比较测量','残差 / 输入 RMS 得到比较 SNR','在分析环带测量两残差相关性','失败或缺失时记未知，不证明非透镜'),
('解释限制','名义 SNR 不含搜索试验校正','未完整计入像素协方差和前景拟合误差','模型一致性支持稳健性，不证明透镜物理'),
]),
('03 搜索与历史候选审计',[
('多阈值信号持久性','基础层：2, 2.58, 3.34, 4.31','5.57, 7.19, 9.29, 12；可自适应变换','threshold_stability 与逐层掩膜'),
('SEP 参数网格一致性','nthresh：16 / 32 / 64','deblend_cont：0.01 / .00215 / .00046 / .0001','合并重复检测，保存网格存活次数'),
('残差结构共识','consensus_residual_structures','形成 segments、分割、footprint、seed mask','原始 SEP 结果保留为 trace，非直接等级'),
('几何路线搜索','中心偏移、轴比与位置角网格','默认最多 5 条路线；在残差上寻找支持','路线只是提案，不填充弧、不合成信号'),
('按路线分区 SEP 像素','partition_sep_segments_by_routes','区分 sep_only 与路线支持的像素区域','过小的碎片只保留在 trace 掩膜'),
('路线局部恢复候选','在路线掩膜内运行恢复检测','保留真实像素、候选来源和关联路径','origin=path_recovery；不自动升级透镜'),
('候选测量与关联','质心 / 半径 / 角度 / 面积 / 积分 SNR','阈值持久性 / 比较模型占据率 / 核心重叠','生成 model_candidates 与候选系统关联'),
('历史结构候选审计','audit_structure_candidates','记录 promoted / hold / trace 等判定','检查数据、模型、持久性和核心敏感性'),
('历史 promoted → 原始结构','primary_candidate / recovery_candidate','提取像素协方差、主轴与轴比','测量切向偏差、径向宽度及角度覆盖'),
('保留全部检测过程','分割、候选掩膜、路线与审计掩膜','正式 S 结构与未晋升候选均可追踪','历史审计不是后续规范化证据的硬门槛'),
]),
('04 原始 Observation 与补测',[
('原始场景几何','结构数量、半径离散 CV、角度并集','平均切向偏差与切向结构比例','这些统计对应原始结构集合'),
('前景与替代解释代理量','前景形状、核心重叠、残差比例','连通性、潮汐不对称、亮度峰、坏像素','代理量不能直接诊断结构的物理起源'),
('保存诊断与原始 Observation','残差、SNR、分割、候选与路线诊断','质量 flags、WCS、参数和 limitations','ring_candidates 为兼容字段，当前不生成'),
('完整候选像素交接','attach_candidate_review','逐候选保存 mask key / SHA / 测量','关联历史 formal_structure_id；缺测也记录'),
('PSF 尺度补测','由正 PSF 二阶矩估计 Gaussian 等效宽度','不是直接测得的半高全宽','用于保守碎片分组和多波段匹配'),
('弧链测量','measure_arc_chains：候选真实像素并集','记录成员、积分通量、SNR 和局部形态','弧段间的空隙不填补'),
('有符号整路线孔径','包括负值和低于阈值的真实像素','signed flux / nominal SNR / 负像素比例','诊断而非经过搜索校准的透镜检测'),
('多波段共同 PSF 孔径','读取可用 VIS / Y / J / H 通量及噪声','PSF 匹配、局部控制、去前景污染孔径','缺波段或低质量信息保持 unknown'),
('颜色关系质量分层','可靠层默认每波段 SNR ≥ 3，共同波段 ≥ 3','提示层 SNR ≥ 2.5，共同波段 ≥ 2','灰度放大偏移、系统底噪与污染检查'),
('多波段关系输出','候选间共同源颜色关系；候选与前景比较','弱颜色关系只提供提示，不直接晋升或否决','PSF / 本征形态起源诊断仍标记 unknown'),
]),
('05 统一证据准备',[
('建立 ExpertState 并深拷贝原始证据','source_observation 保持原样','prepare_evidence：deterministic_evidence_union_v2','固定 ReasoningConfig；计算输入 SHA'),
('旧 Observation 兼容分支','没有完整 handoff 时保留旧结构','不重构缺失像素；写入限制说明','重复 candidate ID 会报错并中止'),
('逐 handoff 候选重新审核','必须存在像素测量；通量 / SNR 有限','默认 pixel_count ≥ 5；SNR ≥ 4；flux > 0','按 mask SHA 去除重复像素候选'),
('纳入或排除均记录原因','保留 formal ID 或使用 REVIEW_candidateID','旧 route / isolation / 占据率不单独否决','未代表于交接表的历史结构仍保留'),
('弧链替换碎片','chain flux > 0 且 SNR ≥ 4 时可替换成员','所有成员必须存在于 handoff','只用测量像素并集；无缺口填充'),
('结构唯一性检查','重复规范化 structure ID 会报错','排序得到唯一 canonical structure 集合','这是测量准入，不是分类投票'),
('重新计算规范化场景','全部结构有 5° occupied angle bins 才计算','角度并集 × 5°，不累加重叠角度','重算半径 CV / 平均偏差 / 切向比例'),
('场景几何缺失分支','空集合或像素角度信息不足 → scene=None','保留结构级测量并声明限制','不沿用不匹配旧掩膜的场景统计'),
('形成共享判断输入','规范化 Observation 进入规则和模型','更新诊断中的等级证据概览','原始与规范化两份证据分别保存'),
]),
('06 唯一规则综合判断',[
('建立可引用事实档案','质量、模型、结构、场景、多波段、比较','每条事实关联 Observation 或 measurement ID','记录可靠性、相关性与未使用信息'),
('可用性及信号真实性','usable 且无 foreground_center_ambiguous','finite_fraction ≥ .99；背景和残差 RMS > 0','正通量且达到名义 SNR 条件 → credible'),
('识别形态并保留风险','切向弧 / 径向结构 / 弯曲结构 / 紧致像','核心穿越 + 径向宽 + 非切向 → 前景盘风险','风险结构不删除，但不单独作为弧证据'),
('系统级构型假设','单稳健弧、fold 对、cusp 组、径向+切向','紧致多像、cross quad、弯曲部分环','共同半径 + 切向 + 覆盖构成传统场景通路'),
('PSF 尺度物理像分组假设','完整链接聚类限制近邻碎片吞并','不改变像素掩膜，不将碎片数当像数','cross quad 使用分组后的四扇区几何'),
('clear_system 的实现条件','任一受支持构型可形成 clear_system','同时要求正通量 / 最低名义 SNR 支持','cross quad 还要求每组有比较模型正信号'),
('颜色只绑定可检验构型','比较被提出为同源多像的候选对','一致 / 部分一致 / 张力 / 混合 / 冲突','可靠同源颜色冲突可限制 A；缺测不否决'),
('比较五种解释','透镜；本征形态；相互作用或并合','前景扣除或 PSF 伪影；噪声或投影源','部分起源测试未实现，只能保留 unknown'),
('关键冲突与缺失信息','相关候选在比较残差中显著负通量 → 冲突','低占据率或模型失败本身不是非透镜证据','缺失信息、处理风险、相关证据分别记录'),
('规则分类：按顺序选择一次','不可用 → insufficient_for_grading','清晰 + 比较支持 + 无关键冲突 → A','清晰且强替代解释或符号冲突 → C'),
('后续分类分支','清晰但可靠颜色冲突或支持不足 → B','可信 + 透镜特异性关系 → B','弱 / 不完整透镜线索 → C；其余可用 → X'),
('保留边界复核与推理轨迹','A 类影子几何条件及复核建议单独保存','不改写正式等级；无加权分数与概率','official expert score 保持空值'),
]),
('07 审计与 DeepSeek 评审',[
('DecisionAudit：确定性完整性检查','事实引用、替代解释覆盖、缺失信息误用','重复 / 相关证据、未验证候选决定性使用','从原始证据重放并比较规则结果'),
('审计结果与 backend 开关','失败会限制规则 A/B/C 发布；不是物理否证','ExpertWorkflow 仍按顺序继续到可选模型','rules 跳过模型；deepseek 进入下列步骤'),
('匿名化模型输入投影','规范化结构、场景、质量、多波段关系','仅保留当前结构/章节引用的原子测量','排除身份、路径、官方标签和规则等级'),
('加载知识与允许引用 schema','几何、弧、颜色、本征形态、伪影、并合','枚举可用且非空的 scalar JSON pointers','PROMPT_VERSION：lens-verification-v5'),
('调用假设生成 API','默认 deepseek-v4-pro；temperature=0','JSON 输出；thinking disabled；timeout 120s','max_tokens 6144；仅保存可见结构化内容'),
('第一阶段输出约束','每个结构必须恰有一条解释及自己的引用','同时包含 lens / non_lens 假设，ID 唯一','记录支持、反证及尚未测得的信息'),
('程序验证引用与结构覆盖','JSON schema；路径存在且非 null/空值','数值和类型一致；引用值由程序取原测量','检查所有规范化结构都被单独解释'),
('失败修复预算（两阶段共享）','验证失败且尚有预算 → 返回错误再调用','最多一次修复；正常 2 次，最多 3 次调用','不是无限对话；服务错误进入失败结果'),
('第二阶段裁决 API','输入同份证据 + 已验证假设 + 验证报告','输出类别、处置、优先假设、理由及限制','同一模型两角色，不是独立专家投票'),
('第二阶段输出验证','优先假设 ID 存在；类别与假设 family 一致','A/B accept；C defer；X reject；不足 defer','data_quality.usable=false 必须弃权'),
('模型成功或失败结果','成功：保存 vote；事后比较规则是否同意','失败：assessment=null，保留规则与错误状态','余额不足 HTTP 402 不等于模型分类错误'),
('验证能力边界','这里只验证引用 / 数值 / 覆盖 / 协议一致性','不验证质量模型、源重建或透镜方程','未实现模型请求工具补测的自动闭环'),
]),
('08 报告 保存 UI 与评估',[
('ReportWriter 汇总','规则证据、类别、审计状态与解释','模型评审单独展示；不平均、不覆盖规则','原始/规范化/模型输入相互可追溯'),
('规则发布字段','审计通过的 A/B/C 可发布相应等级','X 单独记录非透镜投票；不足/失败为 unknown','UI 分组仍依据规则结果'),
('持久化核心结果','source_observation.json / reasoning_observation.json','reasoning.json / summary.json / report.md','可选 llm_input.json / llm_assessment.json'),
('保存复现信息','输入 SHA、代码快照、依赖与配置','模型、提示/知识/输入 SHA、请求/用量/阶段','不保存 API key；温度零不保证模型确定性'),
('CLI 退出状态','0：审计通过且请求的模型完成或未请求','2：规则审计失败（优先于模型失败）','3：审计通过但请求的模型失败'),
('UI 回读与展示','接受 0 / 3 的已完成记录并显示失败状态','规则等级卡、模型等级及理由和限制','文件面板、诊断图与官方参考等级对照'),
('人工复核反馈','查看残差 / 结构 / 模型分歧后修改掩膜','回到前景复核入口 → 全流程重跑','目前由人类触发，不是自动工具闭环'),
('evaluate 事后引入标签','按 id_str 匹配官方 CSV，检查重复对象','分别统计规则/模型与 A/B 等级的一致情况','标签不回流到本次判断'),
('评估分母与结果可用性','全部目标、有效分类、覆盖率、混淆矩阵','失败/未请求记录为 unavailable','10 个 A/B 样本不含 C/X，存在选择偏差'),
('当前试验状态与科学边界','规则等级一致：4/10；非透镜检出率指标','最新版模型 10 次均 HTTP 402，无有效分类','尚不能宣称人类水平或测量污染率'),
])]

CW=530; GAP=70; LEFT=45; TOP=225; NH=147; STEP=175
W=LEFT*2+len(groups)*CW+(len(groups)-1)*GAP
H=TOP+max(len(ns) for _,ns in groups)*STEP+175
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img"><title>强透镜智能体当前实际执行流程详细图</title><desc>八个阶段，包含主要执行步骤、条件、失败分支、证据边界与输出。黑色箭头表示阶段内顺序；蓝色箭头表示跨阶段衔接；虚线表示人工重跑。不是每个底层数值循环的调用图。</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#475569"/></marker></defs><rect width="100%" height="100%" fill="#f8fafc"/>']
def text(x,y,s,size=17,color='#334155',weight='normal'):
    parts.append(f'<text x="{x}" y="{y}" font-family="PingFang SC,Microsoft YaHei,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')
text(45,53,'强引力透镜智能体 当前实际执行流程详细图',34,'#0f172a','bold')
text(45,94,'代码基线：feature/deepseek-judgment · 43347d9 ｜ 按阶段从左至右、阶段内从上至下阅读',21)
text(45,128,'图中包含主要分支和数据约束；分组框内的条件分支只执行适用路径，并非全部同时执行。蓝线连接阶段，红色虚线表示人工重跑。',19)
text(45,161,'当前只进行测量、形态解释和完整性验证；透镜质量拟合、源重建、物理闭合与自动补测仍未实现。默认参数不是经过人类标签校准的概率。',19,'#b45309')
colors=['#eff6ff','#eff6ff','#f0fdfa','#f0fdfa','#faf5ff','#f0fdf4','#fff7ed','#f1f5f9']
for g,(title,ns) in enumerate(groups):
    x=LEFT+g*(CW+GAP)
    parts.append(f'<rect x="{x-12}" y="185" width="{CW+24}" height="{H-300}" rx="18" fill="{colors[g]}" stroke="#cbd5e1"/>')
    text(x+10,215,title,23,'#0f172a','bold')
    for i,n in enumerate(ns):
        y=TOP+i*STEP+15
        fill='#ffffff'
        if any(s in n[0] for s in ['失败','缺失','限制','边界','修复','不足']):fill='#fffbeb'
        parts.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{NH}" rx="10" fill="{fill}" stroke="#94a3b8"/>')
        text(x+16,y+29,f'{g+1:02d}.{i+1:02d}  {n[0]}',21,'#0f172a','bold')
        for j,s in enumerate(n[1:]):text(x+16,y+60+j*27,s,18)
        if i<len(ns)-1:
            parts.append(f'<path d="M{x+CW/2},{y+NH} V{y+STEP}" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    if g<len(groups)-1:
        end=TOP+(len(ns)-1)*STEP+15+NH/2
        nx=x+CW+GAP
        parts.append(f'<path d="M{x+CW},{end} H{x+CW+GAP/2} V{TOP+15+NH/2} H{nx}" stroke="#2563eb" stroke-width="3" fill="none" marker-end="url(#arrow)"/>')
# Explicit entry bypass and human feedback use bottom routing to avoid node text.
by=H-95
xlast=LEFT+7*(CW+GAP)
parts.append(f'<path d="M{xlast+CW},{TOP+6*STEP+15+NH/2} H{W-20} V{by} H20 V{TOP+STEP+15+NH/2} H{LEFT}" stroke="#dc2626" stroke-width="3" stroke-dasharray="10 6" fill="none" marker-end="url(#arrow)"/>')
text(LEFT+CW+150,by-12,'人工反馈：中心或掩膜修改 → 数值提取与判断完整重跑；reason 入口直接从第 05 阶段开始',20,'#b91c1c')
text(45,H-24,'图谱依据：cli.py · vision/pipeline.py · foreground_models.py · supplementary.py · evidence_preparation.py · expert_assessment_agent.py · decision_audit.py · llm_assessment.py · report_writer.py · core/evaluation.py',16)
parts.append('</svg>')
svg=''.join(parts)
ET.fromstring(svg)
(OUT/'01_overall.svg').write_text(svg,encoding='utf-8')
(OUT/'01_overall_detailed.svg').write_text(svg,encoding='utf-8')

# Replace only the canvas placeholder, preserving the download label.
template='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>当前实际执行流程详细图</title><style>body{margin:0;background:#e2e8f0;font-family:-apple-system,"PingFang SC",sans-serif}header{min-height:68px;display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:10px 20px;background:#fff}button,a{padding:8px 13px;color:#0f172a;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;text-decoration:none}#view{height:calc(100vh - 88px);overflow:auto;cursor:grab}#view:active{cursor:grabbing}#canvas svg{display:block;width:100%;height:100%}span{font-size:14px;color:#475569}</style></head><body><header><strong>当前实际执行流程详细图</strong><button onclick="fit()">适应窗口</button><button onclick="zoom(1)">原始尺寸</button><button onclick="zoom(scale/1.25)">缩小</button><button onclick="zoom(scale*1.25)">放大</button><span id="pct"></span><a href="01_overall_detailed.svg" download>下载 SVG</a><a href="SL-agent逻辑分析图谱.html">返回文档</a><span>拖动平移；Ctrl/⌘ + 滚轮缩放</span></header><div id="view"><div id="canvas">__SVG__</div></div><script>
const view=document.getElementById('view'),canvas=document.getElementById('canvas');const W=__WIDTH__,H=__HEIGHT__;let scale=1;
function zoom(s,cx=view.clientWidth/2,cy=view.clientHeight/2){const rx=(view.scrollLeft+cx)/scale,ry=(view.scrollTop+cy)/scale;scale=Math.min(2.5,Math.max(.10,s));canvas.style.width=W*scale+'px';canvas.style.height=H*scale+'px';view.scrollLeft=rx*scale-cx;view.scrollTop=ry*scale-cy;document.getElementById('pct').textContent=Math.round(scale*100)+'%';}
function fit(){zoom(Math.min(view.clientWidth/W,view.clientHeight/H));view.scrollTo(0,0)}
view.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();const r=view.getBoundingClientRect();zoom(scale*Math.exp(-e.deltaY*.005),e.clientX-r.left,e.clientY-r.top)}},{passive:false});
let drag=null;view.onpointerdown=e=>{if(e.pointerType==='mouse'&&e.button===0){drag={x:e.clientX,y:e.clientY,l:view.scrollLeft,t:view.scrollTop};view.setPointerCapture(e.pointerId)}};view.onpointermove=e=>{if(drag){view.scrollLeft=drag.l-(e.clientX-drag.x);view.scrollTop=drag.t-(e.clientY-drag.y)}};view.onpointerup=()=>drag=null;view.onpointercancel=()=>drag=null;zoom(.65);view.scrollTo(0,0);
</script></body></html>'''
(OUT/'当前实际执行流程详细图.html').write_text(template.replace('__SVG__',svg).replace('__WIDTH__',str(W)).replace('__HEIGHT__',str(H)),encoding='utf-8')
doc=OUT/'SL-agent逻辑分析图谱.html'
h=doc.read_text(encoding='utf-8')
if '打开详细流程图，可缩放和平移查看全部步骤' not in h:
    start=h.index('<img src="01_overall.svg"')
    end=h.index('>',start)+1
    h=h[:start]+'<p><a href="当前实际执行流程详细图.html">打开详细流程图，可缩放和平移查看全部步骤</a> · <a href="01_overall_detailed.svg">查看或下载矢量原图</a></p><a href="当前实际执行流程详细图.html"><img src="01_overall.svg" alt="八阶段实际执行流程详细图，点击放大"></a>'+h[end:]
doc.write_text(h,encoding='utf-8')
(OUT/'detailed_flow_nodes.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Generated {sum(len(ns) for _,ns in groups)} detailed nodes in {len(groups)} stages; SVG {W} × {H}.')
