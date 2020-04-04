close all
clear 
clc

format long;
format compact;

addpath('dace', 'gpml-master', 'moead', 'nsga2', 'addGP');

problems       = {'sphere'};
problem_length = length(problems);
Dimension      = 50;
% Popsize        = [50, 100, 200, 300];
% Iteration      = [100, 200];
% totalrun       = 20;
% Dimension      = 10;
Popsize        = 300;
Iteration      = 200;
totalrun       = 2;

for i = 1 : problem_length
    problem = problems{i};
    fprintf('Running on %s...\n', problem);
    for d_idx = 1 : size(Dimension, 2)
        dimension = Dimension(:, d_idx);
        for pop_idx = 1 : size(Popsize, 2)
            popsize = Popsize(:, pop_idx);
            for itr_idx = 1 : size(Iteration, 2)
                iteration = Iteration(:, itr_idx);
                for j = 1 : totalrun
                    sop               = testproblems(problem, dimension);
                    sop.popsize       = popsize;
                    sop.iteration     = iteration;
                    [pop, objs]       = saea(sop);
                    
%                     % Output best objective value every 50 itrs
%                     lb                = 11 * dimension - 1;
%                     lbz               = ceil(lb / 50);
%                     idxout            = lbz*50 : 50 : 1000;
%                     objsout1          = min(objs(1:lb, :));
%                     objsout2          = ones(ceil((1000-lb)/50), 1);
%                     for idxobjout = 1 : ceil((1000-lb)/50)
%                         objsout2(idxobjout) = min(objs(1:idxout(idxobjout), :));
%                     end
%                     objsout           = [objsout1; objsout2];
                    
                    % 'dimension-popsize-itr-run.txt'
                    h                 = '-';
                    filename          = strcat(problem, h, 'd', num2str(dimension), h, 'pop', num2str(popsize), h, 'itr', num2str(iteration), h, num2str(j), '.txt');
                    fp                = fopen(filename, 'a');
                    fprintf(fp, '%d ', objs);
                    fclose(fp);
                end
            end
        end
    end
end