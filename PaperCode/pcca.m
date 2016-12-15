function [WX, WY, PsiX, PsiY, r] = pcca(dimX, dimY, dimZ, covar, inverRegul)

%% sweep input parameter
if nargin < 5
	inverRegul = 0 ;
end

%% configuration
covarXX = covar(1 : dimX, 1 : dimX) + inverRegul * eye(dimX) ;
covarXY = covar(1 : dimX, 1 + dimX : dimY + dimX) ;
covarYX = covarXY' ;
covarYY = covar(1 + dimX : end, 1 + dimX : end) + inverRegul * eye(dimY) ;

%% calculate transform matrix WX
[WX, r] = eig(covarXX \ covarXY / covarYY * covarYX) ;
%% calculate correlation
r = sqrt(r) ;
r = diag(r) ;
% [r, index] = sort(r, 'descend') ;
[r, index] = sort(real(r), 'descend') ;
WX = WX(:, index) ;

%% calculate transform matrix WY
WY = covarYY \ covarYX * WX ;
WY = bsxfun(@times, WY, 1 ./ max(r, eps)') ;

%% tailored to dimZ
WX = WX(:, 1 : dimZ) ;
WY = WY(:, 1 : dimZ) ;
r = r(1 : dimZ) ;

%% Maximum Likelihood Estimated transform models
% WX = covarXX * WX * diag(sqrt(r)) ;
% WY = covarYY * WY * diag(sqrt(r)) ;
%% enforce real matrix
WX = real(covarXX * WX * diag(sqrt(r))) ;
WY = real(covarYY * WY * diag(sqrt(r))) ;

%% Maximum Likelihood Estimated noise models
PsiX = covarXX - WX * WX' ;
PsiY = covarYY - WY * WY' ;
