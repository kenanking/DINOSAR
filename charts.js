/* Progressive enhancement: numerical HTML tables remain the only data source. */
(() => {
  const root = document.querySelector('#results');
  const title = root.querySelector('h2');
  const description = root.querySelector('.section-heading > p:last-child');
  const specs = {
    frozen: ['Frozen features','Frozen features, compared directly','Frozen backbone · k-NN (k = 5) or linear probing · Accuracy (%)',4],
    recognition: ['Recognition','Recognition across acquisition conditions','ATRNet-STAR · Full fine-tuning · 50 epochs · Accuracy (%)',0],
    fewshot: ['Few-shot','More from fewer labels','SAR-ACD · Mean ± standard deviation over five seeds · Accuracy (%)',2],
    detection: ['Detection','Object detection, compared directly','SARDet-100K · Faster R-CNN · 12 epochs · 800 × 800',0],
    segmentation: ['Segmentation','Dense prediction, class by class','AIR-PolSAR-Seg v1.0 · UPerNet · 72 epochs · Mean of three runs',0],
    retrieval: ['Retrieval','Image–text alignment, compared directly','SARVLM · 5,000 evaluation pairs · Unified CLIP fine-tuning · Recall (%)',6]
  };
  const NS = 'http://www.w3.org/2000/svg';
  const green = '#176a4b', sgreen = '#4f9377', gray = '#6e7b86';
  function node(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function svgNode(tag, attrs, text) {
    const el = document.createElementNS(NS,tag);
    Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k,v));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function scaleFor(values) {
    const lo = Math.min(...values.map(v=>v.value-v.error));
    const hi = Math.max(...values.map(v=>v.value+v.error));
    const step = [.1,.2,.5,1,2,5,10,20,25].find(s=>s >= (hi-lo)/6) || 25;
    const min = Math.max(0,Math.floor((lo-.001)/step)*step);
    const max = Math.min(100,Math.ceil((hi+.001)/step)*step);
    return {min,max,step};
  }
  const fmt = value => Number(value.toFixed(2)).toString();
  const renderMath = el => { if (window.renderMathInElement) renderMathInElement(el, { delimiters: [{ left: '\\(', right: '\\)', display: false }] }); };
  // Uniform metric-button width: equal shares when the row fits, otherwise every
  // button keeps the widest button's natural width and the row scrolls.
  function equalizeOptions(scroll) {
    const widths = [...scroll.children].map(o => {
      const clone = o.cloneNode(true);
      // Drop the cloned radio: re-inserting a same-named checked input into the
      // document would uncheck the live one via radio-group exclusivity.
      const clonedInput = clone.querySelector('input');
      if (clonedInput) clonedInput.remove();
      clone.style.cssText = 'position:fixed;inset:0 auto auto 0;visibility:hidden;pointer-events:none;flex:none;max-width:none;contain:layout style';
      document.body.append(clone);
      const w = clone.getBoundingClientRect().width;
      clone.remove();
      return w;
    });
    if (!widths.length) return;
    const width = `${Math.ceil(Math.max(...widths))}px`;
    if (scroll.style.getPropertyValue('--opt-w') !== width) scroll.style.setProperty('--opt-w', width);
  }
  const tabs = node('div','result-tabs'); tabs.setAttribute('role','tablist'); tabs.setAttribute('aria-label','Evaluation task');
  root.querySelector('.section-heading').after(tabs);
  const panels = [...root.querySelectorAll('.result-panel')];
  const renderers = new Map();
  panels.forEach(panel => {
    const [label,heading,protocol,initial] = specs[panel.id];
    const columns = [...panel.querySelectorAll('thead th')].slice(1).map(n=>n.textContent);
    const rows = [...panel.querySelectorAll('tbody tr')].map(tr=>({name:tr.cells[0].textContent,values:[...tr.cells].slice(1).map(td=>{
      const [value,error=0] = td.textContent.split('±').map(Number);return {value,error,label:td.textContent};
    })}));
    const tab = node('button','result-tab'); tab.type='button'; tab.id=`tab-${panel.id}`;
    tab.setAttribute('role','tab');tab.setAttribute('aria-controls',panel.id);tabs.append(tab);
    panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',tab.id);panel.tabIndex=0;
    tab.addEventListener('click',()=>activate(panel));
    tab.addEventListener('keydown',event=>{
      const index=panels.indexOf(panel);let target;
      if(event.key==='ArrowRight')target=(index+1)%panels.length;
      if(event.key==='ArrowLeft')target=(index+panels.length-1)%panels.length;
      if(event.key==='Home')target=0;if(event.key==='End')target=panels.length-1;
      if(target===undefined)return;event.preventDefault();activate(panels[target]);tabs.children[target].focus();
    });
    // Card face: DINOSAR-B headline score on the default metric, plus its margin over the strongest reference baseline.
    const headline=rows.find(r=>r.name==='DINOSAR-B').values[initial];
    const topBaseline=rows.filter(r=>!r.name.startsWith('DINOSAR')&&r.name!=='Random-crop DINO').sort((a,b)=>b.values[initial].value-a.values[initial].value)[0];
    const margin=headline.value-topBaseline.values[initial].value;
    const cardName=node('span','tab-name',label);
    cardName.append(node('span','tab-metric',columns[initial]));
    tab.append(cardName,node('span','tab-value',headline.label),node('span',`tab-delta${margin>0?' is-up':margin<0?' is-down':''}`,margin===0?`matches ${topBaseline.name}`:`${margin>0?'+':'−'}${Number(Math.abs(margin).toFixed(2))} vs ${topBaseline.name}`));
    let metric=initial;
    const ui = panel.querySelector('.chart-ui'); ui.hidden=false;
    const plot = node('div','rank-plot');
    const axisNote = node('p','rank-axis-note');
    const finding = node('p','result-finding'); finding.setAttribute('role','status');
    const controls = node('div','metric-controls');
    controls.setAttribute('role','group');controls.setAttribute('aria-label','Metric');
    const metricScroll=node('div','metric-scroll');metricScroll.tabIndex=0;metricScroll.setAttribute('aria-label','Metric options');
    controls.append(metricScroll);
    columns.forEach((col,index)=>{
      const label=node('label','metric-option');const input=node('input');input.type='radio';input.name=`metric-${panel.id}`;input.value=index;input.checked=index===metric;
      label.append(input,node('span','',col));metricScroll.append(label);
      input.addEventListener('change',()=>{metric=index;draw();reveal(metricScroll,label);});
      input.addEventListener('focus',()=>reveal(metricScroll,label));
    });
    equalizeOptions(metricScroll);
    ui.append(controls,plot,axisNote,finding);
    const details=panel.querySelector('details');details.open=false;details.querySelector('summary').textContent='View full table';
    const note=node('p','scale-note','Point positions use the labeled detail scale. All scores are shown directly; higher is better.');
    panel.append(note);
    function draw() {
      const values=rows.map(row=>row.values[metric]);const {min,max,step}=scaleFor(values);
      const sorted=[...rows].sort((a,b)=>b.values[metric].value-a.values[metric].value);
      const header=node('div','rank-header');header.append(node('span','rank-model-heading','Model'));
      const axis=node('div','rank-axis');header.append(axis,node('span','rank-score-heading',columns[metric]));plot.replaceChildren(header);
      // Each row has a responsive plot and a separate, aligned number column.
      const plotWidth = Math.max(120,axis.getBoundingClientRect().width);
      const x=v=>12+(v-min)/(max-min)*(plotWidth-24);
      const rowHeight=window.matchMedia('(max-width:580px)').matches?36:54;
      const mid=rowHeight/2;
      const tickStride=Math.max(1,Math.ceil(((max-min)/step)*42/(plotWidth-24)));
      const axisSvg=svgNode('svg',{viewBox:`0 0 ${plotWidth} 34`,width:plotWidth,height:34,'aria-hidden':true});
      for(let tick=min,index=0;tick<=max+step/2;tick+=step,index++) if(index%tickStride===0 || tick>=max-step/2) axisSvg.append(svgNode('text',{x:x(tick),y:23,'text-anchor':'middle'},fmt(tick)));
      axis.append(axisSvg);
      sorted.forEach(row=>{
        const v=row.values[metric], ours=row.name==='DINOSAR-B', oursS=row.name==='DINOSAR-S';
        const line=node('div',`rank-row${ours?' is-ours':oursS?' is-ours-s':''}`);
        const name=node('div','rank-name',row.name);
        if(row.name.includes('SAR-1M')) name.append(node('span','rank-control','Pretraining corpus control'));
        const graph=node('div','rank-track');
        const svg=svgNode('svg',{viewBox:`0 0 ${plotWidth} ${rowHeight}`,width:plotWidth,height:rowHeight,role:'img','aria-label':`${row.name}, ${columns[metric]}: ${v.label}`});
        svg.append(svgNode('title',{},`${row.name} — ${columns[metric]}: ${v.label}`));
        for(let tick=min;tick<=max+step/2;tick+=step) svg.append(svgNode('line',{x1:x(tick),x2:x(tick),y1:0,y2:rowHeight,stroke:'#dce1d8','stroke-dasharray':'2 3'}));
        svg.append(svgNode('line',{x1:12,x2:plotWidth-12,y1:mid,y2:mid,stroke:ours?'#176a4b60':oursS?'#176a4b30':'#c8cecc','stroke-width':1}));
        const dotColor=ours?green:oursS?sgreen:gray;
        if(v.error){
          svg.append(svgNode('line',{x1:x(v.value-v.error),x2:x(v.value+v.error),y1:mid,y2:mid,stroke:dotColor,'stroke-width':2}));
          [v.value-v.error,v.value+v.error].forEach(edge=>svg.append(svgNode('line',{x1:x(edge),x2:x(edge),y1:mid-6,y2:mid+6,stroke:dotColor,'stroke-width':2})));
        }
        const dot=oursS?svgNode('circle',{cx:x(v.value),cy:mid,r:5.5,fill:'#fff',stroke:green,'stroke-width':2.5}):svgNode('circle',{cx:x(v.value),cy:mid,r:6,fill:ours?green:gray});svg.append(dot);
        graph.append(svg);line.append(name,graph,node('div','rank-value',v.label));plot.append(line);
      });
      const unit = panel.id==='detection'?'COCO AP':panel.id==='segmentation'?'IoU (%)':'Accuracy (%)';
      axisNote.textContent=`${panel.id==='retrieval'?'Recall (%)':unit} · detail view (${fmt(min)}–${fmt(max)})${panel.id==='fewshot'?' · error bars: ±1 standard deviation':''}`;
      const ours=rows.find(r=>r.name==='DINOSAR-B');
      const baseline=rows.filter(r=>!r.name.startsWith('DINOSAR') && r.name!=='Random-crop DINO').sort((a,b)=>b.values[metric].value-a.values[metric].value)[0];
      const delta=ours.values[metric].value-baseline.values[metric].value;
      finding.replaceChildren(node('strong','', 'DINOSAR-B '), document.createTextNode(delta===0?`matches ${baseline.name}`:`${delta>0?'exceeds':'trails'} ${baseline.name} by `));
      if(delta!==0) finding.append(node('strong','',`${Math.abs(delta).toFixed(2)} ${panel.id==='detection'?'AP':'points'}`));
      finding.append(document.createTextNode(`${panel.id==='detection' && metric===0 ? '' : ` on ${columns[metric]}`} under this protocol.`));
    }
    // Optional small multiples preserve curves without superimposing close results.
    if(panel.id==='fewshot'||panel.id==='retrieval') {
      const trends=node('details','trend-details');trends.append(node('summary','',panel.id==='fewshot'?'View learning curves':'View recall curves'));
      const trendGrid=node('div','trend-grid');trends.append(trendGrid);panel.insertBefore(trends,note);
      function drawTrends(){
        trendGrid.replaceChildren();
        const retrieval=panel.id==='retrieval', groups=retrieval?[0,3]:[0];
        groups.forEach(offset=>rows.forEach(row=>{
          const figure=node('figure','trend-figure');const caption=node('figcaption','',`${row.name}${retrieval?offset?' · Text → image':' · Image → text':''}`);figure.append(caption);trendGrid.append(figure);
          const width=Math.max(250,figure.clientWidth),height=205;
          const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,width,height,role:'img','aria-label':caption.textContent});
          const xs=retrieval?[1,5,10]:[10,20,40], vals=row.values.slice(offset,offset+3);
          const x=v=>40+(v-xs[0])/(xs[2]-xs[0])*(width-92), y=v=>160-v/(retrieval?50:100)*125;
          [0,retrieval?25:50,retrieval?50:100].forEach(t=>{svg.append(svgNode('line',{x1:40,x2:width-52,y1:y(t),y2:y(t),stroke:'#dce1d8'}),svgNode('text',{x:30,y:y(t)+4,'text-anchor':'end'},t));});
          const color=row.name==='DINOSAR-B'?green:row.name==='DINOSAR-S'?sgreen:gray;
          svg.append(svgNode('polyline',{points:vals.map((v,i)=>`${x(xs[i])},${y(v.value)}`).join(' '),fill:'none',stroke:color,'stroke-width':2}));
          vals.forEach((v,i)=>{
            if(v.error) svg.append(svgNode('line',{x1:x(xs[i]),x2:x(xs[i]),y1:y(v.value-v.error),y2:y(v.value+v.error),stroke:color,'stroke-width':2}));
            svg.append(svgNode('circle',{cx:x(xs[i]),cy:y(v.value),r:4,fill:color}),svgNode('text',{x:x(xs[i]),y:y(v.value+v.error)-9,'text-anchor':'middle'},fmt(v.value)),svgNode('text',{x:x(xs[i]),y:183,'text-anchor':'middle'},xs[i]));
          });
          svg.append(svgNode('text',{x:width/2,y:203,'text-anchor':'middle'},retrieval?'K retrieved candidates':'Labeled examples per class'));figure.append(svg);
        }));
      }
      trends.addEventListener('toggle',()=>{if(trends.open)drawTrends();});
      renderers.set(panel,()=>{draw();if(trends.open)drawTrends();});
    } else renderers.set(panel,draw);

  });
  function setupAblation(panel) {
    const points=[...panel.querySelectorAll('tbody tr')].map(tr=>[...tr.cells].map(td=>Number(td.textContent)));
    const ui=panel.querySelector('.chart-ui');ui.hidden=false;
    let k=1,zoom=false,index=points.length-1;
    const controls=node('div','metric-controls');
    const choices=node('div','metric-scroll');choices.tabIndex=0;choices.setAttribute('aria-label','Ablation evaluation metric');controls.append(choices);
    [1,5].forEach(value=>{
      const label=node('label','metric-option'),input=node('input');input.type='radio';input.name='camc-metric';input.checked=value===1;
      const text=node('span','',`k-NN · \\(k = ${value}\\)${value===1?' (paper)':''}`);
      renderMath(text);
      label.append(input,text);choices.append(label);
      input.addEventListener('change',()=>{k=value;draw();});
    });
    equalizeOptions(choices);
    const rangeLabel=node('fieldset','ablation-window');
    rangeLabel.append(node('legend','','Training window'));
    const rangeOptions=node('div','ablation-window-options');
    [['full','Full','5k–145k'],['late','Late','110k–145k']].forEach(([value,title,steps])=>{
      const label=node('label','ablation-window-option'),input=node('input');
      input.type='radio';input.name='camc-window';input.value=value;input.checked=value==='full';
      const text=node('span','ablation-window-text');text.append(node('strong','',title),node('span','',steps));
      label.append(input,text);rangeOptions.append(label);
      input.addEventListener('change',()=>{zoom=value==='late';index=Math.max(index,zoom?21:0);draw();});
    });
    rangeLabel.append(rangeOptions);
    const toolbar=node('div','ablation-toolbar');toolbar.append(controls,rangeLabel);
    const legend=node('div','ablation-legend');legend.append(node('span','camc-key','CAMC'),node('span','random-key','Random crop'));
    const chart=node('div','ablation-plot');
    const sliderLabel=node('label','checkpoint-label','Inspect checkpoint'),slider=node('input');slider.type='range';slider.min=0;slider.max=28;slider.step=1;slider.value=index;sliderLabel.append(slider);
    const readout=node('p','checkpoint-readout');readout.setAttribute('role','status');
    const summary=node('p','ablation-summary');
    ui.append(toolbar,legend,chart,sliderLabel,readout,summary);
    panel.querySelector('details').open=false;
    function draw() {
      const c=k===1?1:4,r=c+1;
      const visible=zoom?points.slice(21):points;
      const mean=col=>points.slice(21).reduce((sum,p)=>sum+p[col],0)/8;
      const width=Math.max(260,chart.clientWidth),height=width<580?280:350;
      const left=42,right=width-18,top=24,bottom=height-42;
      const minX=zoom?110000:5000,minY=zoom?65:0,maxY=80;
      const x=v=>left+(v-minX)/(145000-minX)*(right-left),y=v=>bottom-(v-minY)/(maxY-minY)*(bottom-top);
      const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,width,height,role:'img','aria-label':`CAMC versus random crop, k=${k}, accuracy from ${minY} to 80 percent, ${minX/1000}k to 145k training steps`});
      const add=(tag,attrs,text)=>{const el=svgNode(tag,attrs,text);svg.append(el);return el;};
      if(!zoom)add('rect',{x:x(110000),y:top,width:x(145000)-x(110000),height:bottom-top,fill:'#e8eee4'});
      for(let t=minY;t<=80;t+=zoom?5:20){add('line',{x1:left,x2:right,y1:y(t),y2:y(t),stroke:'#dce1d8'});add('text',{x:left-8,y:y(t)+4,'text-anchor':'end'},t);}
      const ticks=zoom?[110000,120000,130000,145000]:[5000,40000,75000,110000,145000];
      ticks.forEach(t=>add('text',{x:x(t),y:bottom+20,'text-anchor':'middle'},`${t/1000}k`));
      add('text',{x:left,y:14},'Top-1 accuracy (%)');add('text',{x:(left+right)/2,y:height-3,'text-anchor':'middle'},'Pretraining steps');
      if(zoom)[[c,green,'CAMC'],[r,gray,'Random crop']].forEach(([column,color,name])=>{
        const m=mean(column);
        add('line',{x1:left,x2:right,y1:y(m),y2:y(m),stroke:color,'stroke-width':1.5,'stroke-dasharray':'2 3'});
        add('text',{x:left+6,y:y(m)-6,style:`fill:${color};font-size:11px`},`${name} mean ${m.toFixed(2)}`);
      });
      [[r,gray,'5 4','Random crop'],[c,green,'','CAMC']].forEach(([column,color,dash,name])=>{
        add('polyline',{points:visible.map(p=>`${x(p[0])},${y(p[column])}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5,'stroke-dasharray':dash});
        visible.forEach(p=>{const dot=add('circle',{cx:x(p[0]),cy:y(p[column]),r:2.4,fill:color});dot.append(svgNode('title',{},`${name} · ${p[0]/1000}k steps · ${p[column].toFixed(2)}%`));});
      });
      const guide=add('line',{x1:0,x2:0,y1:top,y2:bottom,stroke:'#879387','stroke-dasharray':'3 4'});
      const marks=[c,r].map(col=>add('circle',{r:5,fill:col===c?green:gray,stroke:'white','stroke-width':1.5}));
      function inspect(){
        const p=points[index];slider.value=index;slider.setAttribute('aria-valuetext',`${p[0]/1000} thousand training steps`);
        guide.setAttribute('x1',x(p[0]));guide.setAttribute('x2',x(p[0]));marks.forEach((mark,i)=>{mark.setAttribute('cx',x(p[0]));mark.setAttribute('cy',y(p[i===0?c:r]));});
        readout.textContent=`${p[0]/1000}k steps · CAMC ${p[c].toFixed(2)}% · Random crop ${p[r].toFixed(2)}% · Δ +${(p[c]-p[r]).toFixed(2)} pp`;
      }
      slider.min=zoom?21:0;slider.oninput=()=>{index=Number(slider.value);inspect();};
      function inspectPointer(event){
        const rect=svg.getBoundingClientRect(),px=(event.clientX-rect.left)*width/rect.width;
        const step=minX+(px-left)/(right-left)*(145000-minX);
        index=Math.max(zoom?21:0,Math.min(28,Math.round(step/5000)-1));inspect();
      }
      svg.addEventListener('pointermove',inspectPointer);svg.addEventListener('click',inspectPointer);
      chart.replaceChildren(svg);inspect();
      summary.textContent=`110k–145k checkpoint average (\\(k = ${k}\\)): CAMC ${mean(c).toFixed(2)}% vs random crop ${mean(r).toFixed(2)}% — a ${(mean(c)-mean(r)).toFixed(2)} percentage-point gain. ${zoom?'Detail scale: 65–80%.':'Shaded region: the eight checkpoints used for this average.'}`;
      renderMath(summary);
    }
    return draw;
  }

  function reveal(container,child) {
    const outer=container.getBoundingClientRect(),inner=child.getBoundingClientRect();
    if(inner.left<outer.left)container.scrollLeft-=outer.left-inner.left;
    else if(inner.right>outer.right)container.scrollLeft+=inner.right-outer.right;
  }
  function activate(panel) {
    panels.forEach((p,i)=>{const active=p===panel;p.hidden=!active;tabs.children[i].setAttribute('aria-selected',active);tabs.children[i].tabIndex=active?0:-1;});
    title.textContent=specs[panel.id][1];description.textContent=specs[panel.id][2];renderers.get(panel)();
    reveal(tabs,tabs.children[panels.indexOf(panel)]);
    const checkedMetric=panel.querySelector('.metric-option input:checked');
    if(checkedMetric) reveal(panel.querySelector('.metric-scroll'),checkedMetric.parentElement);
  }
  root.classList.add('results-ready');
  const hash=location.hash.slice(1);activate(panels.find(p=>p.id===hash)||panels.find(p=>p.id==='detection'));
  window.addEventListener('hashchange',()=>{const panel=panels.find(p=>p.id===location.hash.slice(1));if(panel)activate(panel);});
  let previousWidth=root.clientWidth;
  new ResizeObserver(()=>{if(previousWidth!==root.clientWidth){previousWidth=root.clientWidth;root.querySelectorAll('.metric-scroll').forEach(equalizeOptions);const panel=panels.find(p=>!p.hidden);renderers.get(panel)();}}).observe(root);
  const ablation=document.querySelector('#camc-ablation');
  const drawAblation=setupAblation(ablation);
  drawAblation();
  let ablationWidth=ablation.clientWidth;
  new ResizeObserver(()=>{if(ablation.clientWidth!==ablationWidth){ablationWidth=ablation.clientWidth;equalizeOptions(ablation.querySelector('.metric-scroll'));drawAblation();}}).observe(ablation);
  // Web fonts (KaTeX) can widen rendered math labels after first layout.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => {
    document.querySelectorAll('.metric-scroll').forEach(equalizeOptions);
  });
})();
