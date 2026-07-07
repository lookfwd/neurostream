import { matmul, transpose, inverse, cholesky, jacobiEigen, eye } from '../src/linalg.js';
import { dwt, idwt } from '../src/wavelet.js';
let fail=0; const ok=(n,c,x='')=>{console.log(`${c?'PASS':'FAIL'}  ${n}${x?'  — '+x:''}`); if(!c)fail++;};
const maxdiff=(a,b)=>{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]-b[i]));return m;};

// inverse: A*inv(A)=I
{
  const n=5, A=new Float64Array(n*n);
  for(let i=0;i<n*n;i++)A[i]=Math.sin(i*1.3)+ (i%(n+1)===0?3:0);
  const Ai=inverse(A,n); const P=matmul(A,Ai,n,n,n);
  ok('inverse A·A⁻¹=I', maxdiff(P,eye(n))<1e-9, maxdiff(P,eye(n)).toExponential(2));
}
// cholesky: L Lᵀ = A for SPD
{
  const n=4; const B=new Float64Array(n*n); for(let i=0;i<n*n;i++)B[i]=Math.cos(i*0.7);
  const Bt=transpose(B,n,n); const A=matmul(B,Bt,n,n,n); // SPD
  for(let i=0;i<n;i++)A[i*n+i]+=1;
  const Lc=cholesky(A,n); const LLt=matmul(Lc,transpose(Lc,n,n),n,n,n);
  ok('cholesky L·Lᵀ=A', maxdiff(LLt,A)<1e-9, maxdiff(LLt,A).toExponential(2));
}
// jacobi: reconstruct symmetric A = V diag(λ) Vᵀ, and orthonormal V
{
  const n=6; let A=new Float64Array(n*n); for(let i=0;i<n;i++)for(let j=i;j<n;j++){const v=Math.sin(i*2.1+j*0.4); A[i*n+j]=v; A[j*n+i]=v;}
  const {values,vectors}=jacobiEigen(A,n);
  const D=new Float64Array(n*n); for(let i=0;i<n;i++)D[i*n+i]=values[i];
  const VD=matmul(vectors,D,n,n,n); const rec=matmul(VD,transpose(vectors,n,n),n,n,n);
  ok('jacobi V·Λ·Vᵀ=A', maxdiff(rec,A)<1e-8, maxdiff(rec,A).toExponential(2));
  const VtV=matmul(transpose(vectors,n,n),vectors,n,n,n);
  ok('jacobi V orthonormal', maxdiff(VtV,eye(n))<1e-9, maxdiff(VtV,eye(n)).toExponential(2));
}
// wavelet perfect reconstruction at several lengths/levels
for(const N of [256,1024]){
  const x=new Float64Array(N); for(let i=0;i<N;i++)x[i]=Math.sin(i*0.05)+0.3*Math.sin(i*0.9)+ (i%37===0?5:0);
  const dec=dwt(x,4); const rec=idwt(dec);
  ok(`wavelet idwt(dwt(x))=x N=${N}`, maxdiff(rec,x)<1e-9, maxdiff(rec,x).toExponential(2));
}
console.log(`\n${fail===0?'MATH OK':fail+' FAILED'}`); process.exit(fail?1:0);
