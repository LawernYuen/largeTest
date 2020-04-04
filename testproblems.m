function problem = testproblems(testname, dimension)
    
    problem = struct('name', [], 'pd', [], 'domain', [], 'func', []);
    switch lower(testname)
        case 'ellipsoid'
            problem = ellipsoid(problem, dimension);
        case 'sphere'
            problem = sphere(problem, dimension);
        case 'step'
            problem = step(problem, dimension);
        case 'ackley'
            problem = ackley(problem, dimension);
        case 'rosenbrock'
            problem = rosenbrock(problem, dimension);
        case 'rastrigin'
            problem = rastrigin(problem, dimension);
        case 'branin'
            problem = branin(problem, dimension);
        case 'griewank'
            problem = griewank(problem, dimension);
        case 'schwefel'
            problem = schwefel(problem, dimension);
        case 'weierstrass'
            problem = weierstrass(problem, dimension);
        case 'schaffers_f7'
            problem = schaffers_f7(problem, dimension);
        case 'six_hump'
            problem = six_hump(problem, dimension);
        otherwise
            error('Undefined test problem name');
    end
        
end

%% Ellipsoid function generator
function p = ellipsoid(p, dim)
    
    p.name   = 'Ellipsoid';
    p.pd     = dim;
    p.domain = [-5.12 * ones(1, dim); 5.12 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Ellipsoid evaluation function
    function y = evaluate(x)
        idx        = 1 : dim;
%         idx_matrix = repmat(idx, size(x, 1), 1);
%         y          = sum(idx_matrix .* x.^2, 2);
        idx = sort(idx, 'descend');
        y = sum(idx .* (x.^2), 2);
    end
end

%% Sphere
function p = sphere(p, dim)

    p.name   = 'Sphere';
    p.pd     = dim;
    p.domain = [-5.12 * ones(1, dim); 5.12 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Sphere evaluation function
    function y = evaluate(x)
        y = sum(x.^2, 2);
    end
end

%% Step
function p = step(p, dim)

    p.name   = 'Step';
    p.pd     = dim;
    p.domain = [-5.12 * ones(1, dim); 5.12 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Step evaluation function
    function y = evaluate(x)
        x_flr = floor(x);
        y     = sum(x_flr.^2, 2);
    end
end

%% Ackley
function p = ackley(p, dim)

    p.name   = 'Ackley';
    p.pd     = dim;
    p.domain = [-5 * ones(1, dim); 5 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Ackley evaluation function
    function y = evaluate(x)
        a = 20 * exp(-0.2 * sqrt(1 / dim * sum(x.^2, 2)));
        b = exp(1 / dim * sum(cos(2 * pi * x), 2));
        y = 20 + exp(1) - a - b;
    end   
end

%% Rosenbrock
function p = rosenbrock(p, dim)

    p.name   = 'Rosenbrock';
    p.pd     = dim;
    p.domain = [-2.048 * ones(1, dim); 2.048 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Rosenbrock evaluation function
    function y = evaluate(x)
        xi  = x(:, 1 : dim-1);
        xi1 = x(:, 2 : dim);
        yi  = 100 * (xi1 - xi.^2) .^2 + (xi - 1) .^2;
        y   = sum(yi, 2);
    end   
end

%% Rastrigin
function p = rastrigin(p, dim)

    p.name   = 'Rastrigin';
    p.pd     = dim;
    p.domain = [-5.12 * ones(1, dim); 5.12 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Rastrigin evaluation function
    function y = evaluate(x)
        yi = x .^2 - 10 * cos(2 * pi * x);
        y  = 10 * dim + sum(yi, 2);
    end   
end

%% Griewank
function p = griewank(p, dim)

    p.name   = 'Griewank';
    p.pd     = dim;
    p.domain = [-50 * ones(1, dim); 50 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Griewank evaluation function
    function y = evaluate(x)
        y1  = sum(x.^2, 2) / 4000;
        idx = 1 : dim;
        y2  = prod(cos(x./sqrt(idx)), 2);
        y   = y1 - y2 + 1;
    end   
end

%% Schwefel
function p = schwefel(p, dim)

    p.name   = 'Schwefel';
    p.pd     = dim;
    p.domain = [-500 * ones(1, dim); 500 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Schwefel evaluation function
    function y = evaluate(x)
        y1 = x .* sin(sqrt(abs(x)));
        y  = 418.9829 * dim - sum(y1, 2);
    end   
end

%% Weierstrass
function p = weierstrass(p, dim)

    p.name   = 'Weierstrass';
    p.pd     = dim;
    p.domain = [-50 * ones(1, dim); 50 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Weierstrass evaluation function
    function y = evaluate(x)
        k  = 0:20;
        a  = 0.5.^k;
        b  = 3.^k;
%         y1 = bsxfun(@times, x+0.5, reshape(b,1,1,numel(b)));
%         y2 = cos(2 * pi * y1);
%         y3 = bsxfun(@times, y2, reshape(a,1,1,[]));
%         y4 = sum(sum(y3, 3), 2);
%         y5 = a .* cos(pi * b);
%         y6 = dim * sum(y5, 2);
%         y  = y4 - y6;
        y1 = 0;
        for i = 1 : numel(a)
            y1 = y1 + a(i) * cos(2*pi*b(i)*(x+0.5));
        end
        y2 = sum(y1, 2);
        y3 = a .* cos(pi * b);
        y4 = dim * sum(y3, 2);
        y  = y2 - y4;
    end
end

%% Schaffers F7
function p = schaffers_f7(p, dim)

    p.name   = 'Schaffers_F7';
    p.pd     = dim;
    p.domain = [-30 * ones(1, dim); 30 * ones(1, dim)];
    p.func   = @evaluate;
    
    % Schaffers F7 evaluation function
    function y = evaluate(x)
        x1 = x(:, 1:end-1);
        x2 = x(:, 2:end);
        z  = sqrt(x1.^2 + x2.^2);
        y1 = sqrt(z) + sqrt(z) .* (sin(50*z.^0.2).^2);
        y2 = sum(y1, 2) / (dim-1);
        y  = y2 .^ 2;
    end
end

%% Six-Hump Camel function generator
function p = six_hump(p, dim)

    p.name   = 'Six_Hump_Camel';
    p.pd     = dim;
    p.domain = [-3.0, -2.0; 3.0, 2.0];
    p.func   = @evaluate;
    
    % Six-Hump Camel evaluation function
    function y = evaluate(x)
        x1 = x(:, 1);
        x2 = x(:, 2);
        y1 = 4 - 2.1*x1.^2 + x1.^4/3;
        y2 = (4*x2.^2 - 4) .* x2.^2;
        y  = y1.*x1.^2 + x1.*x2 + y2;
    end
end

%% Branin function generator
function p = branin(p, dim)

    p.name   = 'Branin';
    p.pd     = dim;
    p.domain = [-5.0, 0.0; 10.0, 15.0];
    p.func   = @evaluate;
    
    % Branin evaluation function
    function y = evaluate(x)
        a = 1; b = 5.1 / (4 * pi * pi); c = 5 / pi; d = 6; h = 10;
        ff = 1 / (8 * pi);
        x1 = x(:, 1);
        x2 = x(:, 2);
        y  = a .* (x2 - b .* x1.^2 + c .* x1 - d).^2 + ...
            h .* (1 - ff) .* cos(x1) + h;        
    end
end