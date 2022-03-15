% Main function, it is almost the same as function nmf.m in Nonnegative Matrix 
% Factorization Algorithms Toolbox except we enforce L1 norm square constraints 
% on W and H, and further utilize sparseness structure to make algorithm more 
% memory efficient.
% Reference:
%
%    [1] Jingu Kim and Haesun Park.
%        Fast Nonnegative Matrix Factorization: An Active-set-like Method And Comparisons.
%        SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
%
function [W,H,iter,REC]=cosnmf(A,k,varargin)
    % parse parameters
    params = inputParser;
    params.addParamValue('tol'           ,1e-3      ,@(x) isscalar(x) & x > 0);
    params.addParamValue('min_iter'      ,20        ,@(x) isscalar(x) & x > 0);
    params.addParamValue('max_iter'      ,500       ,@(x) isscalar(x) & x > 0);
    params.addParamValue('max_time'      ,1e6       ,@(x) isscalar(x) & x > 0);
    params.addParamValue('init'          ,struct([]),@(x) isstruct(x));
    params.addParamValue('reg_w'         ,[0 0]     ,@(x) isvector(x) & length(x) == 2);
    params.addParamValue('reg_h'         ,[0 0]     ,@(x) isvector(x) & length(x) == 2);
    params.parse(varargin{:});

    [m,n] = size(A);
    par = params.Results;
    par.m = m;
    par.n = n;
    par.k = k;


    % initialize
    if isempty(par.init)
        W = sprand(m,k,0.01); H = sprand(k,n,0.01);
    else
        W = par.init.W; H = par.init.H;
    end

    REC = struct([]);

    clear('init');
    init.norm_A      = norm(A,'fro'); 
    init.norm_W      = norm(W,'fro');
    init.norm_H      = norm(H,'fro');
    init.baseObj     = getObj((init.norm_A)^2,W,H,par);

            
    [gradW,gradH]    = getGradient(A,W,H,par);
    init.normGr_W    = norm(gradW,'fro');
    init.normGr_H    = norm(gradH,'fro');
    init.SC_PGRAD    = getInitCriterion(A,W,H,par,gradW,gradH);

    prev_W = W; prev_H = H;
            
    ver = prepareHIS(A,W,H,prev_W,prev_H,init,par,0,0,gradW,gradH);
    REC(1).init = init;
    REC.HIS = ver;

    display(init);

    tPrev = cputime;

    [W,H,par,val,ver] = anls_bpp_initializer(A,W,H,par);

    if ~isempty(ver)
        tTemp = cputime;
        REC.HIS = saveHIS(1,ver,REC.HIS);
        tPrev = tPrev+(cputime-tTemp);
    end

    REC(1).par = par;
    REC.start_time = datestr(now);
    display(par);

    tStart = cputime; tTotal = 0;
    
    initSC = getInitCriterion(A,W,H,par);

    SCconv = 0; SC_COUNT = 3;

    for iter=1:par.max_iter

        [W,H,gradW,gradH,val] = anls_bpp_iterSolver(A,W,H,iter,par,val);

        elapsed = cputime-tPrev;
        tTotal = tTotal + elapsed;

        clear('ver');
        ver = prepareHIS(A,W,H,prev_W,prev_H,init,par,iter,elapsed,gradW,gradH);

        ver = anls_bpp_iterLogger(ver,par,val,W,H,prev_W,prev_H);
        REC.HIS = saveHIS(iter+1,ver,REC.HIS);

        prev_W = W; prev_H = H;
        display(ver)
        tPrev = cputime;

        if (iter > par.min_iter)
            if (tTotal > par.max_time)
                break;
            else
                SC = getStopCriterion(A,W,H,par,gradW,gradH);
                if (SC/initSC <= par.tol)
                    SCconv = SCconv + 1;
                    if (SCconv >= SC_COUNT)
                        break;
                    end
                else
                    SCconv = 0;
                end
            end
        end
    end
    [m,n]=size(A);
    [W,H]=normalize_by_W(W,H);
    
    final.elapsed_total = sum(REC.HIS.elapsed);
    
    final.iterations     = iter;
    sqErr = getSquaredError(A,W,H,init);
    final.relative_error = sqrt(sqErr)/init.norm_A;
    final.relative_obj   = getObj(sqErr,W,H,par)/init.baseObj;
    final.W_density      = length(find(W>0))/(m*k);
    final.H_density      = length(find(H>0))/(n*k);

    REC.final = final;

    REC.finish_time = datestr(now);
    display(final); 
end

% The following stop citerion idea comes from Lin's implentation
% Reference:
%
%    [1] Chih-Jen Lin.
%        Lin, C. J. (2007). Projected gradient methods for nonnegative matrix factorization. 
%        Neural computation, 19(10), pp. 2756-2779, 2007.
%

% Calculate projected gradients
function pGradF = projGradient(F,gradF)
    pGradF = gradF(gradF<0|F>0);
end

function retVal = getInitCriterion(A,W,H,par,gradW,gradH)
    if nargin~=6
        [gradW,gradH] = getGradient(A,W,H,par);
    end
    [m,k]=size(W); [k,n]=size(H); numAll=(m*k)+(k*n);
    retVal = norm([gradW(:); gradH(:)]);
end

% Compute stop criterion based on projected gradient
function retVal = getStopCriterion(A,W,H,par,gradW,gradH)
    if nargin~=6
        [gradW,gradH] = getGradient(A,W,H,par);
    end
            
    pGradW = projGradient(W,gradW);
    pGradH = projGradient(H,gradH);
    pGrad = [pGradW(:); pGradH(:)];
    retVal = norm(pGrad);
end

% The following functions are modified based on the corresponding functions in Nonnegative Matrix 
% Factorization Algorithms Toolbox
% Reference:
%
%    [1] Jingu Kim and Haesun Park.
%        Fast Nonnegative Matrix Factorization: An Active-set-like Method And Comparisons.
%        SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
%
function [W,H,par,val,ver] = anls_bpp_initializer(A,W,H,par)
    H = zeros(size(H));

    ver.turnZr_W  = 0;
    ver.turnZr_H  = 0;
    ver.turnNz_W  = 0;
    ver.turnNz_H  = 0;
    ver.numChol_W = 0;
    ver.numChol_H = 0;
    ver.numEq_W   = 0;
    ver.numEq_H   = 0;
    ver.suc_W     = 0;
    ver.suc_H     = 0;

    val(1).WtA = W'*A;
    val.WtW = W'*W;
end

function [W,H,gradW,gradH,val] = anls_bpp_iterSolver(A,W,H,iter,par,val)
    
    % Add regularization terms on both W and H
    WtW_reg = applyReg(val.WtW,par,par.reg_h);
    [H,temp,suc_H,numChol_H,numEq_H] = nnlsm_blockpivot(WtW_reg,val.WtA,1,H);

    HHt_reg = applyReg(H*H',par,par.reg_w);
    [W,gradW,suc_W,numChol_W,numEq_W] = nnlsm_blockpivot(HHt_reg,H*A',1,W');
    W = W';

    val.WtA = W'*A;
    val.WtW = W'*W;

    gradW = gradW';
    gradH = getGradientOne(val.WtW,val.WtA,H,par.reg_h,par);

    val(1).numChol_W = numChol_W;
    val.numChol_H = numChol_H;
    val.numEq_W = numEq_W;
    val.numEq_H = numEq_H;
    val.suc_W = suc_W;
    val.suc_H = suc_H;
end

function [ver] = anls_bpp_iterLogger(ver,par,val,W,H,prev_W,prev_H)
    
    ver.turnZr_W    = length(find( (prev_W>0) & (W==0) ))/(par.m*par.k);
    ver.turnZr_H    = length(find( (prev_H>0) & (H==0) ))/(par.n*par.k);
    ver.turnNz_W    = length(find( (prev_W==0) & (W>0) ))/(par.m*par.k);
    ver.turnNz_H    = length(find( (prev_H==0) & (H>0) ))/(par.n*par.k);

    ver.numChol_W   = val.numChol_W;
    ver.numChol_H   = val.numChol_H;
    ver.numEq_W     = val.numEq_W;
    ver.numEq_H     = val.numEq_H;
    ver.suc_W       = val.suc_W;
    ver.suc_H       = val.suc_H;
end

% Calculate residual
function sqErr = getSquaredError(A,W,H,init)
    sqErr = max((init.norm_A)^2 - 2*trace(H*(A'*W))+trace((W'*W)*(H*H')),0 );
end

% Calculate objective value
function retVal = getObj(sqErr,W,H,par)
    retVal = 0.5 * sqErr;
    retVal = retVal + par.reg_w(1) * sum(sum(W.*W));
    retVal = retVal + par.reg_w(2) * sum(sum(W,2).^2);
    retVal = retVal + par.reg_h(1) * sum(sum(H.*H));
    retVal = retVal + par.reg_h(2) * sum(sum(H,1).^2);
end

% Add regularization terms
function AtA = applyReg(AtA,par,reg)
    % Frobenius norm regularization
    if reg(1) > 0
        AtA = AtA + 2 * reg(1) * speye(par.k);
    end
    % L1-norm regularization
    if reg(2) > 0
        AtA = AtA + 2 * reg(2) * sparse(ones(par.k,par.k));
    end
end

function [grad] = modifyGradient(grad,X,reg,par)
    if reg(1) > 0
        grad = grad + 2 * reg(1) * X;
    end
    if reg(2) > 0
        grad = grad + 2 * reg(2) * sparse(ones(par.k,par.k)) * X;
    end
end

function [grad] = getGradientOne(AtA,AtB,X,reg,par)
    grad = AtA*X - AtB;
    grad = modifyGradient(grad,X,reg,par);
end

function [gradW,gradH] = getGradient(A,W,H,par)
    HHt = H*H';
    HHt_reg = applyReg(HHt,par,par.reg_w);

    WtW = W'*W;
    WtW_reg = applyReg(WtW,par,par.reg_h);

    gradW = W*HHt_reg - A*H';
    gradH = WtW_reg*H - W'*A;
end

function [W,H,weights] = normalize_by_W(W,H)
    norm2=sqrt(sum(W.^2,1));
    toNormalize = norm2>0;

    if any(toNormalize)
        W(:,toNormalize) = W(:,toNormalize)./repmat(norm2(toNormalize),size(W,1),1);
        H(toNormalize,:) = H(toNormalize,:).*repmat(norm2(toNormalize)',1,size(H,2));
    end

    weights = ones(size(norm2));
    weights(toNormalize) = norm2(toNormalize);
end

% The following utility functions come from Nonnegative Matrix Factorization 
% Algorithms Toolbox
% Reference:
%
%    [1] Jingu Kim and Haesun Park.
%        Fast Nonnegative Matrix Factorization: An Active-set-like Method And Comparisons.
%        SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.
%

% This function prepares information about execution for a experiment purpose
function ver = prepareHIS(A,W,H,prev_W,prev_H,init,par,iter,elapsed,gradW,gradH)
    ver.iter          = iter;
    ver.elapsed       = elapsed;

    sqErr = getSquaredError(A,W,H,init);
    ver.rel_Error     = sqrt(sqErr)/init.norm_A;
    ver.rel_Obj       = getObj(sqErr,W,H,par)/init.baseObj;
    ver.norm_W        = norm(W,'fro');
    ver.norm_H        = norm(H,'fro');

    ver.rel_Change_W  = norm(W-prev_W,'fro')/init.norm_W;
    ver.rel_Change_H  = norm(H-prev_H,'fro')/init.norm_H;
    
    ver.rel_NrPGrad_W = norm(projGradient(W,gradW),'fro')/init.normGr_W;
    ver.rel_NrPGrad_H = norm(projGradient(H,gradH),'fro')/init.normGr_H;
    ver.SC_PGRAD      = getStopCriterion(A,W,H,par,gradW,gradH)/init.SC_PGRAD; 

    ver.density_W     = length(find(W>0))/(par.m*par.k);
    ver.density_H     = length(find(H>0))/(par.n*par.k);
end

% Execution information is collected in HIS variable
function HIS = saveHIS(idx,ver,HIS)
    fldnames = fieldnames(ver);

    for i=1:length(fldnames)
        flname = fldnames{i};
        HIS.(flname)(idx) = ver.(flname);
    end
end


