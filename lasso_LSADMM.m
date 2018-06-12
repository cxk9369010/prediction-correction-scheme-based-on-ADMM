function [z, historz] = lasso_LSADMM(A, b, r,paras)
%%%     Chang Xiaokai, Bai Jianchao, and Sanyang Liu. Optimal linearized symmetric ADMM
%%%    for multi-block separable convex programming. 
%%%    http://www.optimization-online.org/DB HTML/2018/02/6462.html, 2018.

alpha = paras.alpha;
rho = paras.rho;
lambda = paras.lambda;

gamma=1+alpha;
tau = 1.001*(2+gamma)/4;
r = 1.001*rho*r;
r=r*tau;

t_start = tic;
%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
% MAX_ITER = 50;
ABSTOL   = paras.abstol;
RELTOL   = paras.reltol;
%% Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiplz
Atb = A'*b;
%% ADMM solver

x = zeros(m,1);
z = zeros(n,1);
u = zeros(m,1);


for k = 1:MAX_ITER
    
    % x-update
    x=1/(1+rho)*(b+u+rho*A*z);

    
    
    % z-update with relaxation
    zold = z;
   % x_hat=x;
    u = u - alpha* rho*(x - A*z);% SADMM中第�?��multiplier的parameter等于alpha
    
    
    q=-A' * (u-rho*(x-A*z));
    z = shrinkage(z + q/r, lambda / r);
    % u-update
    u = u - rho*(x - A*z);
    
    % diagnostics, reporting, termination checks
    historz.objval(k)  = objective(A, b, lambda, x, z);
    
     historz.r_norm(k)  = norm(x - A*z);
    historz.s_norm(k)  = norm(-rho*A*(z - zold));
    
    historz.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-A*z));
    historz.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(u);

    
    if (historz.r_norm(k) < historz.eps_pri(k) && ...
            historz.s_norm(k) < historz.eps_dual(k))
             historz.iteration=k;
        historz.time=toc(t_start);
        break;
    end
    
end

% if ~QUIET
%     toc(t_start);
% end
end

function p = objective(A, b, lambda, x, z)
% p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
p = ( 1/2*sum((x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L U] = factor(A, rho)
[m, n] = size(A);
if ( m >= n )    % if skinnz
    L = chol( A'*A + rho*speze(n), 'lower' );
else            % if fat
    L = chol( speze(m) + 1/rho*(A*A'), 'lower' );
end

% force matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
end
