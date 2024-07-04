%task0: Neural Network in MATLAB
% Solve an Input-Output Fitting problem with a Neural Network
% Script generated by Neural Fitting app

% This script assumes these variables are defined:
%   bodyfatInputs - input data.
%   bodyfatTargets - target data.

x = bodyfatInputs;
t = bodyfatTargets;

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

%%
%task1  Feedforward multi-layer networks 
% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
%
% This script assumes these variables are defined:
%   glassInputs - input data.
%   glassTargets - target data.

x = glassInputs;
t = glassTargets;

% Choose a Training Function
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

%%
%task 2: Autoencoder
% get the two classes
class1=1;
class2=8;
%load the mnist data set for just two classes
[X,target] = loadMNIST(0,[class1,class2]);

%train an autoencoder
myAutoencoder = trainAutoencoder(transpose(X),2);
%encode differen classes using the encoder obtained
myEncodedata= encode(myAutoencoder,transpose(X));

%plot using plocl 

%renaming label so they are consecutive numbers as request by plotcl
labels= zeros(size(target,1),1);
for i =1:size(target,1)
    if target(i,1)== class1
        labels(i,1)=1;
    else 
        labels(i,1)=2;
    end
end

plotcl(transpose(myEncodedata),labels)
title(['Classes learning: ', num2str(class1), '-' ,num2str(class2)]);
xlabel('Value on neuron 1');
ylabel('Value on neuron 2');
