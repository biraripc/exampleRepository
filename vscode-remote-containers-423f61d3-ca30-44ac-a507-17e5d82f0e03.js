/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

var h=Object.create;var i=Object.defineProperty;var C=Object.getOwnPropertyDescriptor;var x=Object.getOwnPropertyNames;var E=Object.getPrototypeOf,O=Object.prototype.hasOwnProperty;var P=(e,n)=>{for(var t in n)i(e,t,{get:n[t],enumerable:!0})},a=(e,n,t,o)=>{if(n&&typeof n=="object"||typeof n=="function")for(let s of x(n))!O.call(e,s)&&s!==t&&i(e,s,{get:()=>n[s],enumerable:!(o=C(n,s))||o.enumerable});return e};var m=(e,n,t)=>(t=e!=null?h(E(e)):{},a(n||!e||!e.__esModule?i(t,"default",{value:e,enumerable:!0}):t,e)),w=e=>a(i({},"__esModule",{value:!0}),e);var q={};P(q,{getCredential:()=>l});module.exports=w(q);var g=m(require("http")),f=process.env.REMOTE_CONTAINERS_IPC;function l(e){if(e[1]==="list"){d(e,"").catch(console.error);return}let n="";process.stdin.setEncoding("utf8"),process.stdin.on("data",t=>{n+=t,(n===`
`||n.indexOf(`

`,n.length-2)!==-1)&&(process.stdin.pause(),d(e,n).catch(console.error))}),process.stdin.on("end",()=>{d(e,n).catch(console.error)})}async function d(e,n){let t=await S({args:e,stdin:n});t||process.exit(-1);let{stdout:o,stderr:s,code:r}=JSON.parse(t);o&&process.stdout.write(o),s&&process.stderr.write(s),r&&process.exit(r)}function S(e){return new Promise(n=>{let t=JSON.stringify(e);if(!f){n(void 0);return}let s=g.request({socketPath:f,path:"/",method:"POST"},r=>{let p=[];r.setEncoding("utf8"),r.on("data",c=>{p.push(c)}),r.on("error",c=>u("Error in response",c)),r.on("end",()=>{n(p.join(""))})});s.on("error",r=>u("Error in request",r)),s.write(t),s.end()})}function u(...e){console.error("Unable to connect to VS Code Dev Containers extension."),console.error(...e),process.exit(1)}l(process.argv.slice(2));0&&(module.exports={getCredential});
//# sourceMappingURL=remoteContainersCLI.js.map

