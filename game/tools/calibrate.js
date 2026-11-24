const zlib=require('zlib'),fs=require('fs'),F=require('../static/js/flight.js');
const L=require('./lut.js')();
const K=L.meta.knobs;const keys=[...L.keys()];
const out={},apos=[];let n=0;const t0=Date.now();const tw=[],mar=[];
for(const k of keys){ if(Math.random()>0.15)continue;
  const row=L.row(k);const ox=K.ox[+k[0]];
  let a=null;for(let b=0;b<7;b++){const d=F.design(row,ox,{body:b,fin:3,nose:1});if(d.A.fit){a={body:b,fin:3,nose:1};break}}
  if(!a){out.nofit=(out.nofit||0)+1;continue}
  const Dz=F.design(row,ox,a);tw.push(Dz.tw);mar.push(Dz.A.margin);
  const f=F.fly(Dz,{wind:{spd:4,dir:250}});out[f.outcome]=(out[f.outcome]||0)+1;if(f.outcome==='APOGEE')apos.push(f.apo);n++}
const q=a=>{a.sort((x,y)=>x-y);return [0,.1,.25,.5,.75,.9,1].map(p=>+a[Math.min(a.length-1,Math.floor(p*a.length))].toFixed(1))};
console.log(out,'ms/flight',((Date.now()-t0)/n).toFixed(1));console.log('apo',q(apos));console.log('tw',q(tw));console.log('margin',q(mar));
const row=L.row('333330');
for(const fin of [0,1,2,3,6]){const Dz=F.design(row,5,{body:3,fin,nose:1});const f=F.fly(Dz,{wind:{spd:4,dir:250}});console.log('fin',fin,'m',Dz.A.m0.toFixed(1),'mar',Dz.A.margin.toFixed(2),'Le',Dz.A.Le.toFixed(2),Dz.errs,Dz.warns,f.outcome,f.apo,'tr',f.tr.length)}
