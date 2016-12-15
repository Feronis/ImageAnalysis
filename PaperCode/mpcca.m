function [meanV, covarianceV, Psi, weight, Gamma, transform] = mpcca(X, Y, dimZ, numComponents, maxIterations)
%MPCCA apply mixture probabilistic CCA method on multi-view problems
% using EM algorithm
%INPUT:
% featureX (dimX x numSamples) : View 1 
% featureY (dimY x numSamples) : View 2
% dimZ : dimension of shared space for featureZ
% numComponents : number of Gaussian mixtures
% maxIterations : maximum number of iterations of EM (default 100)
% endFlag : termination tolerance (change in probability likelihood) (default 0.0001)
%INTERMEDIATE
% transformX (dimX x dimZ x numComponents) : transformation matrix from featureZ to featureX
% transformY (dimX x dimZ x numComponents) : transformation matrix from featureZ to featureY
% meanX (numComponents x dimX) :  mean for featureX
% meanY (numComponents x dimY) :  mean for featureY
% varNoiseX ( dimX x dimX x numComponents): covariance for noise in channel X
% varNoiseY ( dimY x dimY x numComponents): covariance for noise in channel Y
% Gamma (numSamples x numComponents) : Gamma_{i, k} = Pr{Component k | sample v_i}
%OUTPUT
% meanV (dimV x numComponents) : mean for joint featureV
% covarianceV (dimV x dimV x numComponents) : covariance for joint featureV
% weight (numComponents x 1) : prior for each component
%MODELS
% featureX = transformX * featureZ + meanX + noiseX
% featureY = transformY * featureZ + meanY + noiseY
%Progress Flag
% logLikelihood : log likelihood
%Termination Condition
% Iterates until a likelihood change < tolerance
% or iterations reach maxIterations

%%%%%%%%%%%%%%%%%%%%%%%%
%% control randomness %%
%%%%%%%%%%%%%%%%%%%%%%%%
rng('default') ;

%%%%%%%%%%%%%%%%%%%%
%% Initialization %%
%%%%%%%%%%%%%%%%%%%%
%% checking data validity
if size(X, 2) ~= size(Y, 2)
	error('multi-view features do not have same amount of data') ;
else
	% V = [X ; Y] ;
	numSamples = size([X ; Y], 2) ;
	dimV = size([X ; Y], 1) ;
end
dimX = size(X, 1) ;
dimY = size(Y, 1) ;

%% using k-means to initialize parameters %%
disp('***** applying k-means *****') ;
meanV = vl_kmeans([X ; Y], numComponents) ;
covarianceV = zeros(dimX + dimY, dimX + dimY, numComponents) ;
for indexComponent = 1 : numComponents
	%% component-wise mean
	V_k = bsxfun(@minus, [X ; Y], meanV(:, indexComponent)) ;
	%% component-wise covariance matrix
	covarianceV(:, :, indexComponent) = V_k * V_k' / numSamples ;
end
weight = ones(1, numComponents) ;
Psi = zeros(dimV, numComponents) ;

%% set of parameters for E-step
meanX = meanV([1 : dimX], :) ;
meanY = meanV([1 + dimX : end], :) ;
%% using probabilistic canonical correlation analysis to initialize parameters %%
transformX = zeros(dimX, dimZ, numComponents) ;
transformY = zeros(dimY, dimZ, numComponents) ;
transform = zeros(dimX + dimY, dimZ, numComponents) ;
covarianceX = zeros(dimX, dimX, numComponents) ;
covarianceY = zeros(dimY, dimY, numComponents) ;
%% initial covariance of X and Y given Z and transform matrix
disp('***** applying probabilistic canonical correlation modelling *****') ;
for indexComponent = 1 : numComponents
	[transformX(:, :, indexComponent), transformY(:, :, indexComponent), ...
	covarianceX(:, :, indexComponent), covarianceY(:, :, indexComponent), relation] = ...
	pcca(dimX, dimY, dimZ, covarianceV(:, :, indexComponent)) ;
	
	transformV_k = [transformX(:, :, indexComponent) ; transformY(:, :, indexComponent)] ;
	covarianceV(:, :, indexComponent) = transformV_k * transformV_k' + ...
										blkdiag(covarianceX(:, :, indexComponent), covarianceY(:, :, indexComponent)) ;
end

gmModel = gmdistribution(meanV', covarianceV, weight) ;
[Gamma, NlogLikelihood] = posterior(gmModel, [X ; Y]') ; 				
%%%%%%%%%%%%%%%%%%%%%%%
%% EM ALGORITHM LOOP %%
%%%%%%%%%%%%%%%%%%%%%%%
for indexIterate = 1 : maxIterations
    fprintf('Iteration#%03d: logLikelihood = %d\n', indexIterate - 1, -NlogLikelihood) ;

	%% backup data
	initCovarianceV = covarianceV ;
	initTransformX = transformX ;
	initTransformY = transformY ;

	%% update prior
	weight = sum(Gamma, 1) ;
	
	% for indexComponent = 1 : numComponents
	parfor indexComponent = 1 : numComponents
	% uncomment to speed up program
		% fprintf('\t EM iteration: %d\n', indexComponent) ;
		%% separately loop for each component %%
		
        %%%%%%%%%%%%
		%% E-step %%
		%%%%%%%%%%%%
		
		%% mean
		meanX(:, indexComponent) = X * Gamma(:, indexComponent) / weight(indexComponent);
		meanY(:, indexComponent) = Y * Gamma(:, indexComponent) / weight(indexComponent);
		% center X
		centerX = bsxfun(@minus, X, meanX(:, indexComponent)) ;
		centerY = bsxfun(@minus, Y, meanY(:, indexComponent)) ;
		
		%%%%%%%%%%%%
		%% M-step %%
		%%%%%%%%%%%%
		
		%% transform matrix for each component
		transformX_k = transformX(:, :, indexComponent) ;
		transformY_k = transformY(:, :, indexComponent) ;

		%%%%%%%%%%%%%%%
		%% posterior %%
		%%%%%%%%%%%%%%%

		% regulation increment
		inverseRegul = 0 ;
		% Var(Z_k) : covariance of latent variable z for k-th component
		covarianceZ_k = eye(dimZ) - ...
			[transformX_k' transformY_k'] / ...
			(covarianceV(:, :, indexComponent) + inverseRegul * eye(dimV)) * ...
			[transformX_k ; transformY_k] ;			
		% E(Z_k) : latent Z samples, dimZ x numSamples
        EZ_k = [transformX_k' transformY_k'] / covarianceV(:, : , indexComponent) * [centerX ; centerY] ;
        % E(Z_k Z_k') : correlation
		correlationZ_k = weight(indexComponent) * covarianceZ_k + bsxfun(@times, EZ_k, Gamma(:, indexComponent)') * EZ_k' ;

		% intermediate variables
		transformX_k = bsxfun(@times, centerX, Gamma(:, indexComponent)') * EZ_k' / correlationZ_k ;
		transformY_k = bsxfun(@times, centerY, Gamma(:, indexComponent)') * EZ_k' / correlationZ_k ;		
		% W^x_k : transform matrix
		transformX(:, :, indexComponent) = transformX_k ;
		% W^y_k : transform matrix
		transformY(:, :, indexComponent) = transformY_k ;

		% intermediate variables
		decoupleX = centerX - transformX_k * EZ_k ;
		decoupleY = centerY - transformY_k * EZ_k ;	
		% Psi^x_k : covariance for noise model of x
		covarianceX_k = transformX_k * covarianceZ_k * transformX_k' + ...
						bsxfun(@times, decoupleX, Gamma(:, indexComponent)') * decoupleX' / weight(indexComponent) ;
		% Psi^y_k : covariance for noise model of y
		covarianceY_k = transformY_k * covarianceZ_k * transformY_k' + ...
						bsxfun(@times, decoupleY, Gamma(:, indexComponent)') * decoupleY' / weight(indexComponent) ;
		
		%%%%%%%%%%%%%%%%%%
		%% Diagonalized %%
		%%%%%%%%%%%%%%%%%%
		covarianceX_k = diag(diag(covarianceX_k)) ;
		covarianceY_k = diag(diag(covarianceY_k)) ;
		Psi(:, indexComponent) = [diag(covarianceX_k) ; diag(covarianceY_k)] ;
		
		% intermediate variable
		transformV_k = [transformX_k ; transformY_k] ;
		% update transform matrix
		transform(:, :, indexComponent) = transformV_k ;
		% Sigma_V
		covarianceV(:, :, indexComponent) = transformV_k * transformV_k' + blkdiag(covarianceX_k, covarianceY_k) ;
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%% singularity processing %%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% assure symmetry
        covarianceV(:, :, indexComponent) = (covarianceV(:, :, indexComponent) + covarianceV(:, :, indexComponent)' ) / 2 ;
		% assure positive definite
		[mapKernel, err] = chol(covarianceV(:, :, indexComponent)) ;
		% recover from good data
		if err
			disp('Singularity! Recovering ...') ;
			transformX(:, :, indexComponent) = initTransformX(:, :, indexComponent) ;
			transformY(:, :, indexComponent) = initTransformY(:, :, indexComponent) ;
			covarianceV(:, :, indexComponent) = initCovarianceV(:, :, indexComponent) ;
			[mapKernel, err] = chol(covarianceV(:, :, indexComponent)) ;
		end
		
		%%%%%%%%%%%%%%
		%% tracking %%
		%%%%%%%%%%%%%%
        % finalize prior
        weight(indexComponent) = weight(indexComponent) / numSamples ;
		% tentative logLikelihood matrix for indexComponent
		Gamma(:, indexComponent) = ...
				log(weight(indexComponent)) - 0.5 * 2 * sum(log(diag(mapKernel))) - dimV * log(2 * pi) / 2 - ...
				sum((mapKernel' \ [centerX ; centerY]) .^ 2, 1)' / 2 ;
		% try to avoid transposing large matrix
	end
	%% M-step %%
	meanV = [meanX ; meanY] ;
   	%% save power %%
	maxFactor = max(Gamma, [], 2) ;
	% recover to exponential
	Gamma = exp(bsxfun(@minus, Gamma, maxFactor)) ;
	% logLikelihood
	NlogLikelihood = - sum(log(sum(Gamma, 2)) + maxFactor) ;
	% posterior
	Gamma = bsxfun(@rdivide, Gamma, sum(Gamma, 2)) ;
end
