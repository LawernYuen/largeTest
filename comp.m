startup;
meanfunc  = @meanConst;
covfunc   = @covSEiso;
likfunc   = @likGauss;
inf       = @infGaussLik;

meanfuncd = @regpoly0;
covfuncd  = @corrgauss;

problem   = 'ellipsoid';
dimension = 2;
problem   = testproblems(problem, dimension);
pop       = randompoint(problem, 100, 'lhs');
objs      = problem.func(pop);

hyp.mean = 0.5;
hyp.cov  = [1; 1];
sn       = 0.1; 
hyp.lik  = log(sn);
hyp      = minimize(hyp, @gp, -200, inf, meanfunc, covfunc, likfunc, pop, objs);

theta = ones(1, problem.pd);
lob   = 0.001 * ones(1, problem.pd);
upb   = 1000 * ones(1, problem.pd);
model = dacefit(pop, objs, meanfuncd, covfuncd, theta, lob, upb);

switch dimension
    case 1
        x         = linspace(0, 6, 61)';
        ygp       = gp(hyp, inf, meanfunc, covfunc, likfunc, pop, objs, x);
        ydace     = predictor(x, model);
        subplot(1,2,1); scatter(x,ygp);
        subplot(1,2,2); scatter(x,ydace);        
    case 2
        x         = [linspace(0, 6, 61)', linspace(0, 6, 61)'];
        ygp       = gp(hyp, inf, meanfunc, covfunc, likfunc, pop, objs, x);
        ydace     = predictor(x, model);
        a = [x(:,1),x(:,2),ygp];
        b = [x(:,1),x(:,2),ydace];
        subplot(1,2,1); mesh(a);
        subplot(1,2,2); mesh(b);
end