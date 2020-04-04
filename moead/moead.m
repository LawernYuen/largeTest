function [parent_pop, parent_objs] = moead(problem, model_hyp, db_pop, db_objs, f_min)
    % This is the main procedure of MOEA/D
    
    %global variable definition.
    global params idealpoint itrCounter meanfunc covfunc likfunc infe nFEs;
    
    %% Set the algorithms parameters.
    %Set up the initial setting for the MOEA/D.
    objDim = 2;
    idealp = ones(1, objDim) * inf;
    
    %the default values for the parameters.
    params.nr        = 2;
    params.niche     = 5;    
    params.delta     = 0.9;
    params.dmethod   = 'i_te';    

    [subproblems, neighbour] = init_weights(params.popsize, params.niche, objDim);
    %initial the subproblem's initital state.
    parent_pop = randompoint(problem, params.popsize, 'random');
    switch params.model
        case 'vanilla'
            parent_objs = problem.func(parent_pop);
        case 'gp'
            [obj, s] = gp(model_hyp, infe, meanfunc, covfunc, likfunc, db_pop, db_objs, parent_pop);
            parent_objs  = [obj, -s];
        case 'kriging'
            [obj, s] = predictor(parent_pop, model_hyp);
            parent_objs  = [obj, -s];
        otherwise
            error('Undefined surrogate methods!')
    end
        
    % Find the ideal point
    idealpoint = min(idealp, min(parent_objs));

    %% Main precedure
    itrCounter = 1;
    while ~terminate(itrCounter)
        [parent_pop, parent_objs] = evolve(parent_pop, parent_objs, subproblems, neighbour, problem, params, model_hyp, db_pop, db_objs);

        fprintf('Dimension: %d :: FE: %d :: iteration %d finished :: Best Fitness = %f\n', problem.pd, nFEs, itrCounter, f_min);
        itrCounter = itrCounter + 1;
    end
end

function [parent_pop, parent_objs] = evolve(parent_pop, parent_objs, subproblems, neighbour, problem, params, model_hyp, db_pop, db_objs)
    global idealpoint meanfunc covfunc likfunc infe;
    
    for i = 1 : params.popsize
        
        if rand < params.delta
            matingindex = neighbour(i, :);
        else
            matingindex = 1 : params.popsize;
        end
            
        % New point generation using genetic operations
        ind = genetic_op(parent_pop, i, problem.domain, params, matingindex);
        switch params.model
            case 'vanilla'
                obj = problem.func(ind);
            case 'gp'
                [mean, s] = gp(model_hyp, infe, meanfunc, covfunc, likfunc, db_pop, db_objs, ind);
                obj  = [mean, -s];
            case 'kriging'
                [mean, ~, s, ~] = predictor(ind, model_hyp);
                obj  = [mean, -s];
            otherwise
                error('Undefined surrogate methods!')
        end
        
        % Update the ideal point
        idealpoint = min(idealpoint, obj);
        
        % Update neighbours
        [parent_pop, parent_objs] = update(parent_pop, parent_objs, subproblems, matingindex, ind, obj, params, idealpoint);
        
        clear ind obj matingindex;
    end
end

function [parent_pop, parent_objs] = update(parent_pop, parent_objs, subproblems, matingindex, ind, ind_obj, params, idealpoint)
    newobj   = subobjective(subproblems(matingindex, :), ind_obj, idealpoint, params.dmethod);
    old_objs = parent_objs(matingindex, :);
    oldobj   = subobjective(subproblems(matingindex, :), old_objs, idealpoint, params.dmethod);
    
    C = newobj < oldobj;
    counter = sum(C);
    newC    = false(size(C));

    if counter <= params.nr
        newC = C;
                
        parent_pop(matingindex(:, newC), :)  = ind(ones(counter, 1), :);
        parent_objs(matingindex(:, newC), :) = ind_obj(ones(counter, 1), :);
    else
        nonzero_ind              = find(C);
        temp                     = randperm(counter); 
        nrInd                    = temp(1 : params.nr);
        newC(nonzero_ind(nrInd)) = 1;
                
        parent_pop(matingindex(:, newC), :)  = ind(ones(params.nr, 1), :);
        parent_objs(matingindex(:, newC), :) = ind_obj(ones(params.nr, 1), :);
    end
    
    clear C newC temp nrInd
end

function y = terminate(itrcounter)
    global params;
    y = itrcounter >= params.maxItr;
end