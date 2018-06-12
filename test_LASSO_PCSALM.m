% L1-regularized least-squares example
%% Generate problem data

randn('seed', 1);
rand('seed',1);

mm = 150;       % number of examples
nn = 500;       % number of features
% pp = 100/n;      % sparsity density
dimension=5;
paras.abstol = 1e-4;
paras.reltol = 1e-2;

aa=[];
alpha= 0;
for ii=1:dimension
    
    m=mm+(ii-1)*150+750;
    mmm(ii)=m;
    n=nn+(ii-1)*500+2500;
    nnn(ii)=n;
    fprintf('&$ %3d \\times %3d$',mmm(ii),nnn(ii));
    p=100/n;
    x0 = sprandn(n,1,p);
    A = randn(m,n);
    A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns

    b = A*x0 + sqrt(0.001)*randn(m,1);

    ATA=A'*A;
    [v,d]=eigs(ATA);
    eig_AA=max(d(:));

    lambda_max = norm( A'*b, 'inf' );
    lambda = 0.1*lambda_max;
    %% Solve problem
    paras.alpha = 0; %% for IDADMM and LSADMM, alpha\in (-1,1)
    paras.rho = 1;   %%  penalty factor 
    paras.lambda = lambda;  %% objective parameter
    paras.step = 1.8;   %% for the correction step of PCSLM 
    paras.tau = 0.5;    %% for the prediction step of PCSLM 
    
    [x2 history2] = lasso_PCSALM(A, b, eig_AA,paras);

    [x3 history3] = lasso_IDSADMM(A, b, eig_AA,paras);
    
    [x4 history4] = lasso_LSADMM(A, b,eig_AA,paras);
    

     kk2(ii)=history2.iteration;
     tt2(ii)=history2.time;
    kk3(ii)=history3.iteration;
    tt3(ii)=history3.time;
    kk4(ii)=history4.iteration;
    tt4(ii)=history4.time;

 %   result_temp=[result_temp kk3(ii), tt3(ii),kk4(ii), tt4(ii), kk4(ii)/kk3(ii),tt4(ii)/tt3(ii)];
%          fprintf(' & %3d(%10.2f) & %3d(%10.2f)&%10.2f(%10.2f)\\\\ \n',...
%          kk4(ii), tt4(ii), kk3(ii),tt3(ii), kk4(ii)/kk3(ii),  tt4(ii)/tt3(ii));
     fprintf(' & %3d(%10.3f)&%3d(%10.3f)&%10.2f(%10.2f)& %3d(%10.3f)&%10.2f(%10.2f)\\\\ \n',...
         kk2(ii), tt2(ii),kk4(ii), tt4(ii),kk2(ii)/kk4(ii),  tt2(ii)/tt4(ii),kk3(ii),tt3(ii), kk2(ii)/kk3(ii),  tt2(ii)/tt3(ii));
    end


    clear A 
    clear b


