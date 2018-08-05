function er_gpu = mlp_tuning(hyp,trainData,trainLabel,validData,validLabel)


hyper_param1=hyp(:,1);
hyper_param2=hyp(:,2);
hyper_param3=hyp(:,3);
hyper_param4=hyp(:,4);
hyper_param5=hyp(:,5);
hyper_param6=hyp(:,6);

rng(0);
no_units                    =roundn(hyper_param1,1);
momentum_variable_hyp       =10.^hyper_param2;
learningRate_variable_hyp   =10.^hyper_param3; 
dropoutFraction_hyp         =hyper_param4;
batchsize_hyp               =round(hyper_param5,1);

no_layers                  =round(hyper_param6);
if no_layers==1
    architecture_nn             =[784 no_units 10];
elseif no_layers==2
    architecture_nn             =[784 no_units no_units 10];
elseif no_layers==3
    architecture_nn             =[784 no_units no_units no_units 10];
end    
  %architecture_nn             =[784 no_units no_units 10]; 
nn                          = nnsetup(architecture_nn);
%nn                          = nnsetup([784 1200 1200 1200 10]);
nn.output                   = 'softmax'; % output function: softmax | sigm | linear
nn.activation_function      = 'sigm';    % activation func: sigm | tanh_opt | linear
%nn.dropoutFraction          = 0.5;       % Droupout of hidden layers
nn.dropoutFraction          = dropoutFraction_hyp;       % Droupout of hidden layers
nn.inputZeroMaskedFraction  = 0.2;       % input dropout
%nn.inputZeroMaskedFraction  = inputZeroMaskedFraction_hyp;
%nn.weightPenaltyL2         = 1e-6;      % weightdecay
nn.weightMaxL2norm          = 15;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
nn.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
nn.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
nn.errfun                   = @nntest;

opts.plotfun                = @nnplottest;
opts.numepochs              =  1;        %  Number of full sweeps through data
opts.momentum_variable      =momentum_variable_hyp;
%opts.momentum_variable      = [linspace(0.5,0.95,1500 ) linspace(0.95,0.95,opts.numepochs -1500)];
%opts.learningRate_variable  =  8.*(linspace(0.998,0.998,opts.numepochs ).^linspace(1,opts.numepochs,opts.numepochs ));
%opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.learningRate_variable  = learningRate_variable_hyp;
%opts.learningRate_variable  = repmat(learningRate_variable_hyp,numepochs_hyp,1)';
opts.plot                   = 0;            % 0 = no plotting, migth speed up calc if epochs run fast
opts.batchsize              = batchsize_hyp;         % Take a mean gradient step over this many samples. GPU note: below 500 is slow on GPU because of memory transfer
opts.ntrainforeval          = 360;         % number of training samples that are copied to the gpu and used to evalute training performance
opts.outputfolder           = 'tests/nns/hinton'; % saves network each 100 epochs and figures after 10. hinton is prefix to the files. 
                                            % nns is the name of a folder
                                            % from where this script is
                                            % called (probably tests/nns)
                                        

tt = tic;
[nn,L,loss]                 = nntrain_gpu(nn, trainData, trainLabel, opts,validData,validLabel); %use nntrain to train on cpu
toc(tt);
[er_gpu, bad]               = nntest(nn, validData, validLabel);    
% fprintf('Error: %f \n',er_gpu);
