function [decomp, paramsGP, numGroups] = preprocessDecomposition(problem, params, paramsGP, addRemainingDims)

  % First set the Decomposition
  if ~isfield(paramsGP, 'decompStrategy') | isempty(paramsGP.decompStrategy)
    paramsGP.decompStrategy = 'partialLearn';
  end

  if ~exist('addRemainingDims', 'var')
    addRemainingDims = true;
  end

  if addRemainingDims
    numGroups = ceil(problem.pd/params.d);
    numRemDims = problem.pd - params.d * (numGroups-1);
    if numRemDims == 0
      addRemainingDims = false;
    end
  else
    numGroups = floor(problem.pd/params.d);
  end

  % Determine the decomposition accordingly.
  if strcmp(paramsGP.decompStrategy, 'stoch1')
    decomp.dMax = params.d;

  elseif params.d == problem.pd
    % This is full (naive) BO
    paramsGP.decompStrategy = 'known';
    decomp = cell(1,1);
    decomp{1} = 1:problem.pd;
    paramsGP.noises = 0 * ones(numGroups, 1);

  elseif strcmp(paramsGP.decompStrategy, 'known')
    decomp = cell(numGroups, 1);
    paramsGP.noises = 0 * ones(numGroups, 1);
    for i = 1:numGroups
      decomp{i} = ( (i-1)*params.d+1 : min(i*params.d, problem.pd) );
    end

  elseif addRemainingDims
    decomp = [ params.d * ones(numGroups-1, 1); numRemDims];
  
  else
    decomp.d = params.d;
    decomp.M = numGroups;

  end

end

