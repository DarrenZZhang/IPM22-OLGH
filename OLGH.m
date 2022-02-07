function [B_train , B_test ] = OLGH (data_our, opt, nbits)
tol = 5e-7; 
[~,nFea] = size(data_our.X); %Ntrain
X  = data_our.X(data_our.indexTrain, :); X=X'; %train
X2 = data_our.X(data_our.indexTest, :); X2 = X2';%test
y  = data_our.label(data_our.indexTrain, :);

% label matrix Y = N x c
if isvector(y)
    Y = sparse(1:length(y), double(y), 1); Y = full(Y');
else
    Y = y';
end
nCls = size(Y,1);

%%%%%%%% initialize %%%%%%
XXT = X*X';
D   = randn(nbits,nCls);
B   = sign(D*Y);

Imat = eye(nFea);
Winit = Imat(:,1:nbits);
[~, L] = TripletLap(X', opt.k);

%-----------training-------------------
for iter=1:opt.Iter_num
    
    Rii = 2*sqrt(sum(Winit.^2,2));
    Rii(Rii==0) = tol;
    R = diag(1./Rii);
    
    D = learn_basis(B, Y, opt.eta);

    B = zeros(size(B));  
    B = sign(Winit'*X + opt.lambda*D*Y);
    
    Winit = (XXT + opt.beta*R + opt.gamma*L)\(X*B');
end


B_train=B'>0;
%--Out-of-Sample------
NT = (X * X' + 1 * eye(size(X, 1))) \ X;
W  = NT*B_train;
B_test=X2'*W>0;
 
end



