const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
function harness(){
 const player={src:'previous',muted:false,pauses:0,loads:0,pause(){this.pauses++;},load(){this.loads++;},removeAttribute(name){delete this[name];}},panel={open:true,hidden:false};
 const context=vm.createContext({$:id=>id==='source-video'?player:panel});
 vm.runInContext(source.slice(source.indexOf('function setSourcePlayback('),source.indexOf('let demoTicket=')),context);
 return {context,player,panel};
}
test('local source playback stays separate, muted and collapsed when changing samples',()=>{
 const h=harness();h.context.setSourcePlayback({original_video:'http://127.0.0.1:1234/known'});
 assert.equal(h.player.pauses,1);assert.equal(h.player.loads,1);assert.equal(h.player.src,'http://127.0.0.1:1234/known');
 assert.equal(h.player.muted,true);assert.equal(h.panel.open,false);assert.equal(h.panel.hidden,false);
});
test('opening a still or ordinary legacy source unloads the previous local video',()=>{
 const h=harness();h.context.setSourcePlayback({kind:'image'});assert.equal(h.player.src,undefined);assert.equal(h.panel.hidden,true);assert.equal(h.player.pauses,1);
});
