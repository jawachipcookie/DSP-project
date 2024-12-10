classdef adam < handle
    properties
        learning_rate
        weight_decay
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m
        v
        t
    end
    
    methods
        function obj = adam()
            obj.m = containers.Map();
            obj.v = containers.Map();
            obj.t = 0;
        end
        
        function step(obj, model, gradients)
            obj.t = obj.t + 1;
            
            for param_name = keys(gradients)
                if ~isKey(obj.m, param_name{1})
                    obj.m(param_name{1}) = zeros(size(gradients(param_name{1})));
                    obj.v(param_name{1}) = zeros(size(gradients(param_name{1})));
                end
                
                grad = gradients(param_name{1});
                if obj.weight_decay > 0
                    grad = grad + obj.weight_decay * model.(param_name{1});
                end
                
                obj.m(param_name{1}) = obj.beta1 * obj.m(param_name{1}) + ...
                    (1 - obj.beta1) * grad;
                obj.v(param_name{1}) = obj.beta2 * obj.v(param_name{1}) + ...
                    (1 - obj.beta2) * grad.^2;
                
                m_hat = obj.m(param_name{1}) / (1 - obj.beta1^obj.t);
                v_hat = obj.v(param_name{1}) / (1 - obj.beta2^obj.t);
                
                model.(param_name{1}) = model.(param_name{1}) - ...
                    obj.learning_rate * m_hat ./ (sqrt(v_hat) + obj.epsilon);
            end
        end
    end
end