
function [db_pop, db_objs] = saea(problem)

    global nFEs params meanfunc covfunc likfunc infe;
    
    % set the algorithm parameters
    [db_pop, db_objs, f_min] = init(problem);
    
    while ~terminate(nFEs)
%         % Training data = initial data + recent data (50%/50%)
%         ini_size = 11 * problem.pd - 1;
%         db_size  = size(db_objs, 1);
%         if db_size > 2 * (ini_size)
%             tr_pop  = [db_pop(1:ini_size, :); db_pop(db_size-ini_size+1:db_size, :)];
%             tr_objs = [db_objs(1:ini_size, :); db_objs(db_size-ini_size+1:db_size, :)];
%         else
%             tr_pop  = db_pop;
%             tr_objs = db_objs;
%         end

        % Training data = recent best db_size data
        db_size  = size(db_pop, 1);
        if db_size > params.db_size
            [~, idx] = sort(db_objs);
            tr_pop  = db_pop(idx(1:params.db_size), :);
            tr_objs = db_objs(idx(1:params.db_size), :);
        else
            tr_pop  = db_pop;
            tr_objs = db_objs;
        end

%         % Training data: 2m <= n <= 4m
%         db_size  = size(db_pop, 1);
%         if db_size > 4 * params.popsize
%             tr_pop  = db_pop(db_size-4*params.popsize+1:db_size, :);
%             tr_objs = db_objs(db_size-4*params.popsize+1:db_size, :);
%         else
%             tr_pop  = db_pop;
%             tr_objs = db_objs;
%         end

%         % Training data: recent db_size
%         db_size  = size(db_pop, 1);
%         if db_size > params.db_size
%             tr_pop  = db_pop(db_size-params.db_size+1:db_size, :);
%             tr_objs = db_objs(db_size-params.db_size+1:db_size, :);
%         else
%             tr_pop  = db_pop;
%             tr_objs = db_objs;
%         end
        
        % build (or re-build) the surrogate model
        model_hyp = model_building(problem, tr_pop, tr_objs);
        
        % Optimise acquisition function
        switch params.optimize
            case 'pso'
                new_sample = pso(problem, model_hyp, tr_pop, tr_objs, f_min);
            case 'de'
                new_sample = de(problem, model_hyp, tr_pop, tr_objs, f_min);
            case 'cmaes'
                new_sample = cmaes(problem, model_hyp, tr_pop, tr_objs, f_min);
            case 'quasi-newton'
                new_sample = quasi_newton(problem, model_hyp, tr_pop, tr_objs, f_min);
            case 'direct'
                new_sample = direct(problem, model_hyp, tr_pop, tr_objs, f_min);
            case 'moead'
                [pop, popobj] = moead(problem, model_hyp, tr_pop, tr_objs, f_min);
            case 'nsga2'
                [pop, popobj] = nsga2(problem, model_hyp, tr_pop, tr_objs, f_min);
        end
        
%         % Landscape (acq - 1d source space)
%         lb = min(problem.domain);
%         ub = max(problem.domain);
%         x = lb:1/50:ub;
%         popx = x.';
%         y_r = problem.func(popx);
%         [yobj, ys] = gp(model_hyp, infe, meanfunc, covfunc, likfunc, db_pop, db_objs, popx);
%         f = [yobj+2*sqrt(ys); flip(yobj-2*sqrt(ys),1)];
%         acq = acquisition_func(f_min, yobj, ys);
%         [~, idx] = min(acq);
%         fill([popx; flip(popx,1)], f, [7 7 7]/8);
%         hold on;
%         plot(popx, yobj); % approximation curve
%         plot(popx, y_r);  % function curve
%         plot(pop, popobj(:,1), '+');
%         plot(popx(idx,:), yobj(idx,:), 'o');
%         pause(0.1);
%         clf;        
        
%         [~, ns_idx] = min(popobj(:,1));
%         new_sample = pop(ns_idx,:);

        pop = pop(1:problem.pd, :);

%         problem.domain(1,:) = min(pop(:))*ones(1,problem.pd);
%         problem.domain(2,:) = max(pop(:))*ones(1,problem.pd);

        new_sample = pop;
        new_sample_obj = problem.func(new_sample);
        nFEs = nFEs + size(new_sample, 1);
        if min(new_sample_obj) < f_min
            f_min = min(new_sample_obj);
        end
        db_pop  = [db_pop; new_sample];
        db_objs = [db_objs; new_sample_obj];
        
%         if (numel(db_objs)-params.db_size)/(2*problem.pd) >= ceil((1001-11*problem.pd)/(2*problem.pd))/5
%             gap = (problem.domain(2,:)-problem.domain(1,:))*0.99 / (2*5);
%             problem.domain(1,:) = problem.domain(1,:) + gap;
%             problem.domain(2,:) = problem.domain(2,:) - gap;
%         end
    end
        
    clear newsample tr_objs tr_pop
    pack
end

%% initialisation process
function [pop, objs, f_min] = init(problem)

    global nFEs params meanfunc covfunc likfunc inf;
    
    % parameter settings
    params.model    = 'addGP'; % model type: GP, Kriging
    params.db_size  = 11 * problem.pd - 1;
    params.popsize  = problem.popsize;  % population size of DE
    params.maxFEs   = 1000;
    params.maxItr   = problem.iteration;  % number of generations of DE
    params.acq_type = 'EI'; % type of the acquisition function
    params.optimize = 'nsga2';
    params.d        = 10;    % Add-GP-UCB with maximum group size d = 4
    params.numItr   = 200;
    
    switch params.model
        case 'gp'
            % parameter settings of GP
            startup;
            meanfunc = @meanConst;
            covfunc  = @covSEiso;
            likfunc  = @likGauss;
            inf      = @infLaplace;
        case 'kriging'
            meanfunc = @regpoly0;
            covfunc  = @corrgauss;
        case 'addGP'
            [decomp, paramsGP] = preprocessDecomposition(problem, params, struct(), true);
            params.decomp = decomp;
            params.paramsGP = paramsGP;
    end
    
    % initialise a population
    pop  = randompoint(problem, params.db_size, 'lhs');
    objs = problem.func(pop);
    nFEs = params.db_size;
    
    % find the current best fitness
    f_min = min(objs);
end

%% model training and hyperparameter estimation
function model = model_building(problem, db_pop, db_objs)
    global meanfunc covfunc likfunc inf params;
    
    if strcmp(params.model, 'vanilla') == 1
        model = [];
    elseif strcmp(params.model, 'gp') == 1
        hyp.mean = 0.5;
        hyp.cov  = [1; 1];
    
        sn      = 0.1; 
        hyp.lik = log(sn);
    
        hyp   = minimize(hyp, @gp, -200, inf, meanfunc, covfunc, likfunc, db_pop, db_objs);
        model = hyp;    % this represents the GP model hyperparameters
    elseif strcmp(params.model, 'kriging') == 1
        theta = ones(1, problem.pd);
        lob   = 0.001 * ones(1, problem.pd);
        upb   = 1000 * ones(1, problem.pd);

        model = dacefit(db_pop, db_objs, meanfunc, covfunc, theta, lob, upb);
    elseif strcmp(params.model, 'addGP') == 1
        [combFuncH, funcHs] = addGPBO(problem, params.paramsGP, params.decomp, db_pop, db_objs, params.numItr);
        model = {combFuncH, funcHs};
    else
        error('Undefined surrogate model!')
    end
    
end

%% termination checking
function y = terminate(nFEs)

    global params;
    y = nFEs >= params.maxFEs;
    
end