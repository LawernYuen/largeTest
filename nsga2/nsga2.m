function [popp, popo] = nsga2(problem, model_hyp, db_pop, db_objs, f_min)

    %global variable definition.
    global params nFEs;
    
    %% Set the algorithms parameters.
    nObj       = 2;
    nVar       = problem.pd;
    lb         = min(problem.domain(:));
    ub         = max(problem.domain(:));
    nPop       = params.popsize;
    
    % SBX & polynomial mutation parameters
    proC = 1;    % crossover probability
    disC = 20;   % distribution index of sbx
    proM = 1;    % expectation of number of bits doing mutation
    disM = 20;   % distribution index of polynomial mutation
    
    % BLX & normally distributed mutation parameters
    pCrossover = 0.7;                         % Crossover Percentage
    nCrossover = 2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)
    pMutation  = 0.4;                         % Mutation Percentage
    nMutation  = round(pMutation*nPop);       % Number of Mutants
    mu         = 0.02;                        % Mutation Rate
    sigma      = 0.1*(ub-lb);                 % Mutation Step Size
    
    % DE operators parameters
    deF  = 0.5;
    deCR = 0.5;

    %% Initialization
    empty_individual.Position         = [];
    empty_individual.Cost             = [];
    empty_individual.Rank             = [];
    empty_individual.DominationSet    = [];
    empty_individual.DominatedCount   = [];
    empty_individual.CrowdingDistance = [];

    pop  = repmat(empty_individual, nPop, 1);
%     inilb = repmat(inilb, params.popsize, 1);
%     iniub = repmat(iniub, params.popsize, 1);
%     popp = unifrnd(inilb, iniub, [nPop, nVar]);
    popp = unifrnd(lb, ub, [nPop, nVar]);
    popo = CostFunction(problem, popp, model_hyp, db_pop, db_objs);
    pop = assignV(popp, popo, pop);
    
    [pop, F] = nonDominatedSort(pop);  % Non-Dominated Sorting
    pop = crowdingdistance(pop, F);    % Calculate Crowding Distance
    [pop, F] = sortpop(pop);           % Sort Population
    
    %% NSGA-II Main Loop
    for itr = 1 : params.maxItr
        
        popPos = getV(pop, problem.pd);

        %% (DE/rand/1)
        idx = randperm(params.popsize);
        idx(idx == 1) = [];
        for i = 2 : params.popsize
            idxtemp = randperm(params.popsize);
            idxtemp(idxtemp == i) = [];
            idx = [idx; idxtemp];
        end
        a = idx(:, 1);
        b = idx(:, 2);
        c = idx(:, 3);

        % Mutation
        newpoint = popPos(a,:) + deF*(popPos(b,:) - popPos(c,:));
        % Crossover
        jrandom             = ceil(rand(params.popsize,1) * problem.pd);
        randomarray         = rand(params.popsize, problem.pd);
        deselect            = randomarray < deCR;
        linearInd           = sub2ind(size(deselect),1:params.popsize,jrandom');
        deselect(linearInd) = true;
        newpoint(~deselect) = popPos(~deselect);

        % repair
        newpoint = max(newpoint, lb);
        newpoint = min(newpoint, ub);

        newobj = CostFunction(problem, newpoint, model_hyp, db_pop, db_objs);
        popnew = repmat(empty_individual, params.popsize, 1);
        popnew = assignV(newpoint, newobj, popnew);
        rng('shuffle')
%         
%         % Merge
%         pop = [pop; popnew];

        %% BLX crossover
        popc = repmat(empty_individual, nCrossover, 1);
        i1 = randi(nPop, [nCrossover/2, 1]);
        i2 = randi(nPop, [nCrossover/2, 1]);
        p1 = popPos(i1, :);
        p2 = popPos(i2, :);
        [popcPos1, popcPos2] = crossover(p1, p2, lb, ub);
        popcPos = [popcPos1; popcPos2];
        popcObj = CostFunction(problem, popcPos, model_hyp, db_pop, db_objs);
        popc = assignV(popcPos, popcObj, popc);

        % Mutation        
        popm = repmat(empty_individual, nMutation, 1);
        i = randi(nPop, [nMutation, 1]);
        p = popPos(i, :);
        popmPos = mutate(p, mu, sigma, lb, ub);
        popmObj = CostFunction(problem, popmPos, model_hyp, db_pop, db_objs);
        popm = assignV(popmPos, popmObj, popm);
        rng('shuffle')
%         
%         % Merge
%         pop = [pop; popc; popm];

        %% SBX & polynomial mutation
        offspring = reproduction(popPos, proC, disC, proM, disM, lb, ub);
        offspringCost = CostFunction(problem, offspring, model_hyp, db_pop, db_objs);
        popr = repmat(empty_individual, size(offspring, 1), 1);
        popr = assignV(offspring, offspringCost, popr);
        rng('shuffle')
% 
%         % Merge
%         pop = [pop; popr];

        %%

        % Merge
        pop = [pop; popnew; popc; popm; popr];

        [pop, F] = nonDominatedSort(pop);  % Non-Dominated Sorting
        pop = crowdingdistance(pop, F);    % Calculate Crowding Distance
        pop = sortpop(pop);                % Sort Population
        pop = pop(1:nPop);                 % Truncate

        [pop, F] = nonDominatedSort(pop);  % Non-Dominated Sorting
        pop = crowdingdistance(pop, F);    % Calculate Crowding Distance`
        pop = sortpop(pop);                % Sort Population
        F1 = pop(F{1});                    % Store F1

       disp(['Dimension = ' num2str(problem.pd) ': FE = ' num2str(nFEs) ': Iteration ' num2str(itr) ': Number of F1 Members = ' num2str(numel(F1)) ': Fmin = ' num2str(f_min)]);
    end

    %% Sort by front, mean
    sortidx = [];
    for i = 1 : numel(F)
        [~, obji] = getV(pop(F{i}), problem.pd);
        [~, idx]  = sort(obji(:, 1));
        sortidx = [sortidx, idx'];
    end
    pop = pop(sortidx);
    [popp, popo] = getV(pop, problem.pd);
    
    %% Visualize PF
%     corrplot(popo, 'varNames',{'mean','variance'});
%     pause(0.1);
%     figname = strcat(problem.name, '-d', num2str(problem.pd), '.png');
%     saveas(gcf, figname);

%     % pareto set, 1d
%     [vpopp, seq] = sort(popp);
%     vpopo = popo(seq, :);
%     rpopo = problem.func(vpopp);
%     f = [vpopo(:,1)+0.1*sqrt(vpopo(:,2)); flip(vpopo(:,1)-0.1*sqrt(vpopo(:,2)),1)];
%     fill([vpopp; flip(vpopp,1)], f, [7 7 7]/8);
%     hold on;
%     plot(vpopp, vpopo(:,1));
%     plot(vpopp, rpopo);

%     % sampling points, 1d
%     x = lb:1/50:ub;
%     popx = x.';
%     yobj2 = CostFunction(problem, popx, model_hyp, db_pop, db_objs);
%     yobj = yobj2(:, 1);
%     ys = yobj2(:, 2);
%     scatter(yobj, ys);
    
    
end

function obj = CostFunction(problem, pop, model_hyp, db_pop, db_objs)

    global params meanfunc covfunc likfunc infe;

    switch params.model
        case 'vanilla'
            obj = problem.func(pop);
        case 'gp'
            [mean, s] = gp(model_hyp, infe, meanfunc, covfunc, likfunc, db_pop, db_objs, pop);
            obj  = [mean, s];
        case 'kriging'
            if size(pop, 1) > 1
                [mean, s] = predictor(pop, model_hyp);
                obj  = [mean, s];
            else
                [mean, ~, s, ~] = predictor(pop, model_hyp);
                obj  = [mean, s];
            end
        case 'addGP'
            add_func = model_hyp{1};
            [mean, s] = add_func(pop);
            obj = [mean, s];
        otherwise
            error('Undefined surrogate methods!')
    end    
end

function [pos, obj] = getV(pop, d)

    pos = ones(numel(pop), d);
    obj = ones(numel(pop), 2);
    for i = 1 : numel(pop)
        pos(i,:) = pop(i).Position;
        obj(i,:) = pop(i).Cost;
    end
    
end

function pop = assignV(pos, obj, pop)

    for i = 1 : size(pos, 1)
        pop(i).Position = pos(i,:);
        pop(i).Cost = obj(i,:);
    end

end
