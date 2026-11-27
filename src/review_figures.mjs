// Recreate the exact review snapshot SVGs; requires Node.js, no packages.
// Snapshot values are from E01–E04/E08 summary JSON and notebooks 14/16 outputs at 8f4a6822b010cc8d75da83d33c3f1be59b43464c.
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
const data = {
  "baseline": [
    [
      "Scratch",
      0.7927831265470489
    ],
    [
      "Feature Extraction",
      0.7678417988807454
    ],
    [
      "Partial Fine-tuning",
      0.8310509390292623
    ],
    [
      "Full Fine-tuning",
      0.8581409847086494
    ]
  ],
  "cm": [
    [
      63,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      160,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      116,
      0,
      0,
      0,
      0,
      39,
      0,
      5,
      0,
      0
    ],
    [
      0,
      0,
      0,
      128,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      2,
      0,
      0,
      0,
      125,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      124,
      0,
      0,
      0,
      0,
      0,
      4
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      160,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      31,
      0,
      0,
      0,
      0,
      101,
      0,
      28,
      0,
      0
    ],
    [
      0,
      0,
      10,
      0,
      0,
      0,
      0,
      2,
      147,
      1,
      0,
      0
    ],
    [
      0,
      0,
      28,
      0,
      0,
      0,
      0,
      29,
      1,
      6,
      0,
      0
    ],
    [
      5,
      0,
      0,
      0,
      9,
      0,
      0,
      0,
      0,
      0,
      50,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      56
    ]
  ],
  "labels": [
    "Y-Exercise",
    "니푸쉬업",
    "덤벨 로우",
    "딥스",
    "레그 레이즈",
    "랫풀다운",
    "로잉머신",
    "데드리프트",
    "바벨 런지",
    "바벨 로우",
    "바이시클",
    "케이블 크런치"
  ],
  "cv": [
    [
      "aspect384",
      0.0283,
      9,
      1,
      1.4331,
      true
    ],
    [
      "중앙 크롭",
      0.0278,
      8,
      2,
      1.0179,
      false
    ],
    [
      "stretch",
      0.0144,
      8,
      0,
      0.9829,
      true
    ]
  ],
  "ds": [
    [
      "Scratch",
      [
        0.6854,
        0.7287,
        0.7535
      ]
    ],
    [
      "Feature Extraction",
      [
        0.7234,
        0.729,
        0.7481
      ]
    ],
    [
      "Full Fine-tuning",
      [
        0.8283,
        0.8393,
        0.8449
      ]
    ]
  ],
  "people": [
    "Z24",
    "Z59",
    "Z83"
  ],
  "clips": 179
};
function makeFigures(d){
const C={bg:"#f6f8fb",white:"#ffffff",ink:"#172b45",muted:"#54677e",line:"#d8e2ec",blue:"#2463a6",teal:"#087e83",orange:"#bb592c",gray:"#91a3b8",purple:"#7967ad"};
const e=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const txt=(x,y,s,size=16,fill=C.ink,weight=400,anchor="start",extra="")=>`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" ${extra}>${e(s)}</text>`;
const rect=(x,y,w,h,fill=C.white,stroke="none",rx=8,extra="")=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const line=(x,y,x2,y2,stroke=C.line,sw=1,extra="")=>`<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
const dot=(x,y,r,fill)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
const wrap=(title,sub,w,h,body,source)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="title desc"><title id="title">${e(title)}</title><desc id="desc">${e(sub)}</desc><rect width="${w}" height="${h}" rx="18" fill="${C.bg}"/><g font-family="Arial, 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans CJK KR', sans-serif">${txt(30,42,title,26,C.ink,700)}${txt(30,67,sub,14,C.muted)}${body}${txt(30,h-17,source,12,C.muted)}</g></svg>\n`;
const n=d.cm.flat().reduce((a,b)=>a+b,0),ok=d.cm.reduce((a,r,i)=>a+r[i],0),mf=d.cm.reduce((a,r,i)=>a+2*r[i]/(r.reduce((a,b)=>a+b,0)+d.cm.reduce((a,b)=>a+b[i],0)),0)/12;
let b=rect(30,87,394,100,C.white,C.line,12)+rect(444,87,406,100,C.white,C.line,12)+txt(50,113,"FINAL TEST  /  MACRO F1",13,C.muted,700)+txt(50,157,mf.toFixed(3),39,C.blue,700)+txt(444+20,113,"FINAL TEST  /  ACCURACY",13,C.muted,700)+txt(464,157,(100*ok/n).toFixed(1)+"%",39,C.teal,700)+txt(243,158,"12개 클래스",14,C.muted)+txt(669,158,`${ok.toLocaleString("en-US")} / ${n.toLocaleString("en-US")}장`,14,C.muted);
b+=txt(30,224,"기본 학습 전략 · Validation Macro F1",19,C.ink,700);
for(let i=0;i<d.baseline.length;i++){let [label,v]=d.baseline[i],y=244+49*i;b+=txt(30,y+19,label,15,C.ink,i===3?700:400)+rect(244,y,490,26,"#e5ecf3","none",5)+rect(244,y,490*v,26,i===3?C.blue:C.gray,"none",5)+txt(844,y+19,v.toFixed(3),17,C.ink,700,"end")}
b+=txt(244,449,"0",12,C.muted)+txt(734,449,"1.0",12,C.muted,400,"end")+txt(30,478,"Test는 최종 선정 Full FT · letterbox 224 · seed 43의 1회 평가 결과.",14,C.muted);
const overview=wrap("피트니스 운동 분류 · 핵심 성과","ResNet18 | 12개 운동 | 동일 분할에서 학습 전략 비교",880,513,b,"출처: experiments/E01~E04/*/summary.json, E08/test_summary.json");
b="";
const boxes=[[30,95,246,96,n.toLocaleString("en-US")+"장","Test 이미지",C.blue],[317,95,246,96,d.clips+"개","클립 (8장씩)",C.teal],[604,95,246,96,d.people.length+"명","독립된 사람 코드",C.orange]];
for(const [x,y,w,h,v,t,col] of boxes)b+=rect(x,y,w,h,C.white,C.line,12)+txt(x+w/2,y+41,v,31,col,700,"middle")+txt(x+w/2,y+69,t,15,C.muted,400,"middle");
for(const x of [282,569])b+=line(x,143,x+29,143,C.gray,2)+`<path d="M${x+23} 137L${x+29} 143L${x+23} 149" fill="none" stroke="${C.gray}" stroke-width="2"/>`;
b+=rect(30,210,820,70,"#fff4e8","none",10)+txt(48,237,"해석 주의",15,C.orange,700)+txt(48,261,"이미지 1,432장을 평가했지만 새로운 사람에 대한 일반화 검증 대상은 3명입니다.",16,C.ink);
const pop=wrap("평가 표본의 실제 구조","이미지 수와 사람 단위 표본 수는 동일한 개념이 아닙니다.",880,317,b,"출처: E08/pred_test.csv (고유 image_id·clip_id·person_id)");
b=txt(177,92,"예측 클래스 →",15,C.muted,700)+txt(30,116,"실제 클래스 ↓",14,C.muted,700);
let x0=177,y0=142,cell=49;
for(let i=0;i<12;i++){
 b+=txt(164,y0+i*cell+30,d.labels[i],14,i===9?C.orange:C.ink,i===9?700:400,"end")+txt(x0+i*cell+cell/2,130,String(i),14,C.muted,700,"middle");
 for(let j=0;j<12;j++){
 const v=d.cm[i][j],row=d.cm[i].reduce((a,b)=>a+b,0),q=v/row;
 b+=rect(x0+j*cell,y0+i*cell,cell-3,cell-3,v?(i===j?C.blue:C.orange):"#e9eef4","none",4,v?`opacity="${(0.15+0.85*q).toFixed(3)}"`:"");
 if(v)b+=txt(x0+j*cell+(cell-3)/2,y0+i*cell+19,Math.round(q*100)+"%",13,q>.60?"#ffffff":C.ink,700,"middle")+txt(x0+j*cell+(cell-3)/2,y0+i*cell+35,String(v),11,q>.60?"#ffffff":C.muted,400,"middle")
 }
 b+=txt(839,y0+i*cell+28,String(d.cm[i].reduce((a,b)=>a+b,0)),13,C.muted,400,"end")
}
b+=txt(839,130,"n",13,C.muted,700,"end");
for(let i=0;i<12;i++){let x=30+(i%3)*273,y=765+Math.floor(i/3)*25;b+=txt(x,y,`${i}  ${d.labels[i]}`,13,C.muted)}
b+=rect(30,872,820,50,"#fff4e8","none",9)+txt(46,902,"바벨 로우: 64장 중 6장 정답(9.4%) · 58장 오답",17,C.orange,700);
const cm=wrap("최종 Test · 클래스별 혼동행렬","각 칸: 실제 클래스 내 비율(위), 이미지 수(아래). 파랑=정답, 주황=오답.",880,955,b,"출처: experiments/E08/pred_test.csv 및 test_summary.json");
b=txt(30,106,"전처리",14,C.muted,700)+txt(251,106,"평균 Δ Macro F1",14,C.muted,700)+txt(568,106,"개선 쌍",14,C.muted,700,"middle")+txt(657,106,"붕괴 fold",14,C.muted,700,"middle")+txt(805,106,"판정",14,C.muted,700,"middle");
for(let i=0;i<d.cv.length;i++){let [name,v,positive,collapse,time,pass]=d.cv[i],y=124+i*66,col=pass?C.teal:C.orange;b+=rect(30,y,820,53,C.white,C.line,8)+txt(45,y+32,name,16,C.ink,700)+rect(251,y+13,205,26,"#e5ecf3","none",5)+rect(251,y+13,205*v/.035,26,col,"none",5)+txt(516,y+33,"+"+v.toFixed(4),15,C.ink,700,"end")+txt(568,y+32,positive+"/9",16,C.ink,700,"middle")+txt(657,y+32,collapse+"/3",16,collapse>=2?C.orange:C.ink,700,"middle")+txt(805,y+32,pass?"채택":"기각",16,col,700,"middle")}
b+=rect(30,333,820,75,"#e8f3f3","none",10)+txt(45,359,"운영 기준",14,C.teal,700)+txt(45,384,"평균 Δ ≥ +0.010 · 6/9쌍 이상 개선 · 2/3 fold 이상 개선 · 붕괴 fold 2개 미만",15,C.ink);
b+=txt(30,437,"aspect384도 fold 1개에서 클래스 붕괴가 발생했습니다. 최종 Test 재평가 결과는 아닙니다.",14,C.muted);
const cv=wrap("전처리 비교 · 성능과 클래스 붕괴","동일 fold·seed의 letterbox 대비 짝 비교(3 folds × 3 seeds)",880,476,b,"출처: notebooks/14_cv_stage2.ipynb 저장 결과");
b="";let lx=88,rx=817,ty=112,by=335;const sy=v=>by-(v-.60)/.30*(by-ty),sx=i=>lx+i*(rx-lx)/2;
for(let v of [.6,.7,.8,.9])b+=line(lx,sy(v),rx,sy(v))+txt(lx-12,sy(v)+5,v.toFixed(2),13,C.muted,400,"end");
for(let i=0;i<3;i++)b+=txt(sx(i),by+25,[25,50,100][i]+"%",15,C.muted,700,"middle");
for(let i=0;i<d.ds.length;i++){let [name,values]=d.ds[i],col=[C.gray,C.purple,C.blue][i];b+=`<polyline points="${values.map((v,j)=>`${sx(j)},${sy(v)}`).join(" ")}" stroke="${col}" stroke-width="3" fill="none"/>`;for(let j=0;j<3;j++){b+=dot(sx(j),sy(values[j]),5,col);let offset=i===0?19:i===1?-12:-13;b+=txt(sx(j)+(j===0?12:j===2?-12:0),sy(values[j])+offset,values[j].toFixed(3),13,i===2?C.blue:C.muted,700,j===0?"start":j===2?"end":"middle")}}
for(let i=0;i<3;i++){let x=[89,282,540][i];b+=line(x,389,x+24,389,[C.gray,C.purple,C.blue][i],3)+dot(x+12,389,4,[C.gray,C.purple,C.blue][i])+txt(x+32,394,d.ds[i][0],14,C.ink,600)}
b+=rect(30,416,820,62,"#e8f3f3","none",10)+txt(46,441,"Full FT 25% > Scratch 100% · 3/3 fold",17,C.teal,700)+txt(46,462,"단, 데이터 양과 학습 업데이트 횟수의 영향을 분리한 인과 실험은 아닙니다.",14,C.muted);
const ds=wrap("데이터 양에 따른 전이학습 효과","사람 단위 3-fold 검증 Macro F1 평균 · seed 42",880,514,b,"출처: notebooks/16_data_size.ipynb 저장 결과");
return {"overview.svg":overview,"test_population.svg":pop,"test_confusion.svg":cm,"cv_preprocessing.svg":cv,"data_size.svg":ds};
}
const dest = fileURLToPath(new URL("../figures/review/", import.meta.url));
mkdirSync(dest, {recursive:true});
for (const [name, svg] of Object.entries(makeFigures(data))) writeFileSync(join(dest,name),svg,"utf8");
