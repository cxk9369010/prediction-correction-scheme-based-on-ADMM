function [z, historz] = lasso_PCSALM(A, b,eig_AA,paras)
%%%   Xiaokai. Chang ¡¤ Sanyang. Liu ¡¤ Zhao Deng,
%%%   A prediction-correction scheme based on Lagrange multiplier
%%%   with indefinite proximal regularization for separable convex programming


tau = paras.tau;
step = paras.step;
rho = paras.rho;
lambda = paras.lambda;


r = 1.001*rho*eig_AA;


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
    
    %% x-update
    x=1/(1+rho)*(b+u+rho*A*z);

    
   %% u-update
    u = u -  rho*(x - A*z);% SADMMä¸­ç¬¬ä¸?¸ªmultiplierçš„parameterç­‰äºŽalpha
    
   %% z-update 
    q=-A' * (u-rho*(x-A*z));
    z_p = shrinkage(z + q/r, lambda / r);
    
    
    %% u-update
    u_p = u + rho*A*(z_p - z);
 %%%%%%%%%%%%%%%%%%%%%%%  Correction step   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    uu_p = u -u_p;
    zz_p = z -z_p;
    
    num1 = norm(uu_p,'fro')^2/rho + r * norm(zz_p,'fro')^2 - dot(A*(zz_p),uu_p);
    
    Q_X_XN = [r*zz_p; uu_p/rho];
 
    alpha_k = step * num1/norm(Q_X_XN,'fro')^2;
    
    z = z - alpha_k * r * zz_p;
    u = u - alpha_k * uu_p/rho;
  %   z = z_p;
  %   u = u_p;

    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
    
    
    % diagnostics, reporting, termination checks
    historz.objval(k)  = objective(A, b, lambda, x, z_p);
    
     historz.r_norm(k)  = norm(x - A*z_p);
    historz.s_norm(k)  = norm(-rho*A*(z_p - z));
    
    historz.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-A*z_p));
    historz.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(u_p);
    

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
