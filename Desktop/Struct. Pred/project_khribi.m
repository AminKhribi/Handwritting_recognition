%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Structured Prediction Project %%%%
%%%%%%%%%%%% AMIN KHRIBI %%%%%%%%%%%%%%

clear all
clear c

%% Multi Class Perceptron

%load data
addpath('svm-struct-matlab-1.1')
disp ( 'loading data...')
path = 'C:\Users\khribia\Desktop\Struct. Pred\letter.data';
data = importdata(path);
%

n_train1 = 100;
n_train2 = 100;

disp ('compute W with binary perceptron..')

for i = 97:122 %double values of characters from a to z
    %trainning set of n_train1 vectores of the right letter
    %and n_train2 of other letters (drawn randomly)
    
    letter_index = find(strcmp(data.textdata(:,2),char(i)));
    data_letter = data.data(letter_index(1:n_train1), 5:end); 
      
    for j=1:n_train2
        id = floor(rand(1)*n_train2);
        while ismember(id, letter_index)||(id==0)
            id = floor(rand(1)*n_train2);
        end
        data_letter = cat(1, data_letter, data.data(id, 5:end));
    end
    data_letter = data_letter';
    labels = zeros(1,n_train1 + n_train2);
    for j=1:(n_train1 + n_train2)
        if (j<=n_train1)
           labels(j) = 1;
        else
           labels(j) = 0;
        end
    end
    
   % binary perceptron: net package in Matlab
   net = newp(data_letter, labels);
   net.trainParam.epochs = 55; 
   net = train (net, data_letter, labels);
   if i==97
       W = net.IW{1};
       b = net.b{1};
   else
      W = cat(1, W, net.IW{1});
      b = cat(1, b, net.b{1});
   %err = length(find(labels~=sim(net, data_a)));
   end
   clear letter_index
   clear data_letter
   clear labels
end
W = W';
b = b';

%error rate before oprimization
tic
err = 0;
for i = 1:size(data.data,1)
   x = data.data(i, 5:end); 
   [~,ybar] = max(x*W + b);
   if strcmp(char(ybar+96), data.textdata(i,2))==0
       err = err + 1;
   end
end
t_perc = toc;
disp (['error before optimization: ', num2str(err/size(data.data,1))])
disp (['execution time: ', num2str(t_perc)])

%optimize W over the whole data set
%!! we could have started with random values for wi i<=26 !!
disp ('optimizing multi class perceptron..')
for i = 1:size(data.data,1)
   x = data.data(i, 5:end); 
   [~,ybar] = max(x*W + b);
   ytrue = data.textdata(i,2);
   if strcmp(char(ybar+96), ytrue)==0
       U = zeros(128, 26);
       for j = 1:26
           U(:,j) = -x;
       end
       ytrue_index = double(char(ytrue)) - 96; 
       U(:,ytrue_index) = x;
       W = W + U;
   end
end

%error rate after optimization
tic
err = 0;
for i = 1:size(data.data,1)
   x = data.data(i, 5:end); 
   [~,ybar] = max(x*W + b);
   if strcmp(char(ybar+96), data.textdata(i,2))==0
       err = err + 1;
   end
end
t_mlp = toc;
disp (['error after optimization: ', num2str(err/size(data.data,1))])
disp (['execution time: ', num2str(t_mlp)])

%% SVM binary classifier 
svmStruct = cell(1,26);

kernel = 'linear'; %choose kernel
disp ('compute binary SVM ..')
for i = 97:122
    %prepare trainning set
    letter_index = find(strcmp(data.textdata(:,2),char(i)));
    data_letter = data.data(letter_index(1:n_train1), 5:end);
      
    for j=1:n_train2
        id = floor(rand(1)*n_train2);
        while ismember(id, letter_index)||(id==0)
            id = floor(rand(1)*n_train2);
        end
        data_letter = cat(1, data_letter, data.data(id, 5:end));
    end
    data_letter = data_letter';
    labels = zeros(1,n_train1 + n_train2);
    for j=1:(n_train1 + n_train2)
        if (j<=n_train1)
           labels(j) = 1;
        else
           labels(j) = 0;
        end
    end
    
   % binary svm
   options = optimset('maxiter',1000); 
   svmStruct{i-96} = svmtrain(data_letter, labels,...
            'kernel_function', kernel ,'quadprog_opts',options);
   %sv{i-96} = svmStruct.SupportVectors;
   %alphaHat(i-96) = svmStruct.Alpha;
   %bias(i-96) = svmStruct.Bias;
   %kfun(i-96) = svmStruct.KernelFunction;
   %kfunargs(i-96) = svmStruct.KernelFunctionArgs;
      
   clear letter_index
   clear data_letter
   clear labels
end

%svm error rate 
disp ('calculating SVM error ..')
tic
err = 0;
for i = 1:size(data.data,1)
   x = data.data(i, 5:end); 
   f = zeros(1, 26);
   for j = 1:26
      %y = sum(kernel(xi,xtest)*alpha) + biais
      %I didn't find a way to get the score out of Matlab SVM package..
      sv = svmStruct{j}.SupportVectors;
      alphaHat = svmStruct{j}.Alpha;
      bias = svmStruct{j}.Bias;
      kfun = svmStruct{j}.KernelFunction;
      kfunargs = svmStruct{j}.KernelFunctionArgs;
      ff = kfun(sv, x, kfunargs{:})'*alphaHat(:) + bias;
      f(j) = ff*-1; 
   end
   [~,ybar] = max(f);
   if strcmp(char(ybar+96), data.textdata(i,2))==0
       err = err + 1;
   end
end
t_svm = toc;
disp (['svm error: ', num2str(err/size(data.data,1))])
disp (['execution time: ', num2str(t_svm)])

%% error correction with HMM

%computation of the transition matrix
disp ('computing transition matrix..')
TransM = zeros(26, 26);
for i = 97:122
   list = find(strcmp(data.textdata(:,2), char(i)));
   for j = 1:size(list, 1)
       next_id = data.data(list(j),1);
       if next_id ~= -1
           k = data.textdata(next_id, 2);
           TransM(double(char(k))-96, i-96) = TransM(double(char(k))-96, i-96) + 1;
       end
   end 
end

%normalization
for i = 1:26
    TransM(:,i) = TransM(:,i)/sum(TransM(:,i));
end

% computation of the emission matrix
classifier = 'MCP';
disp (['computing emission matrix for classifier: ', classifier])
tic
%
EmisM = zeros(26, 26);
for i = 97:122
    
    list = find(strcmp(data.textdata(:,2), char(i)));
    if strcmp(classifier, 'MCP')
        for j = 1:size(list, 1)
            x = data.data(list(j), 5:end);
            [~,ybar] = max(x*W + b); 
            EmisM(ybar, i-96) = EmisM(ybar, i-96) + 1;
        end
    elseif strcmp(classifier, 'SVM')
        for j = 1:size(list, 1)
            x = data.data(list(j), 5:end);
            f = zeros(1, 26);
            for l = 1:26
                sv = svmStruct{l}.SupportVectors;
                alphaHat = svmStruct{l}.Alpha;
                bias = svmStruct{l}.Bias;
                kfun = svmStruct{l}.KernelFunction;
                kfunargs = svmStruct{l}.KernelFunctionArgs;
                ff = kfun(sv, x, kfunargs{:})'*alphaHat(:) + bias;
                f(l) = ff*-1; 
            end
            [~,ybar] = max(f);
            EmisM(ybar, i-96) = EmisM(ybar, i-96) + 1;
        end
    end
end

%normalization
for i = 1:26
    EmisM(:,i) = EmisM(:,i)/sum(EmisM(:,i));
end  
    
t_comp_emis = toc;
disp (['computation time of emmision matrix: ', num2str(t_comp_emis)])

%% recognition test
i = 1;
err = 0;
classifier = 'MCP';
while i<50000
    clear x_test
    clear ytrue
    % compose a test word
    x_test = data.data(i, 5:end);
    ytrue = data.textdata (i, 2);
    next_id = data.data(i,1);
    while next_id ~= -1
        i = i + 1;
        x_test = cat(1, x_test, data.data(next_id, 5:end));
        ytrue = cat(2, ytrue, data.textdata (next_id, 2));
        next_id = data.data(i, 1);
    end

    %run our classifier on letters independantly
    ybar = zeros(1, size(x_test, 1));
    if strcmp(classifier, 'MCP')
        for j = 1:size(x_test, 1)
             x = x_test(j,:);
             [~,ybar(j)] = max(x*W + b); 
        end
    elseif strcmp(classifier, 'SVM')
        for j = 1:size(x_test, 1)
             x = x_test(j,:);
             f = zeros(1, 26);
             for l = 1:26
                 sv = svmStruct{l}.SupportVectors;
                 alphaHat = svmStruct{l}.Alpha;
                 bias = svmStruct{l}.Bias;
                 kfun = svmStruct{l}.KernelFunction;
                 kfunargs = svmStruct{l}.KernelFunctionArgs;
                 ff = kfun(sv, x, kfunargs{:})'*alphaHat(:) + bias;
                 f(l) = ff*-1; 
             end
             [~,ybar(j)] = max(f);
        end
    end
    
    %input into Viterbi algorithm
    pStates = hmmdecode(ybar, TransM, EmisM);
    [~,yhmm] = max(pStates);
    char_yhmm = char(yhmm+96);
    char_class = char(ybar+96);
    %disp (['the true word is: ', char(ytrue)'])
    %disp (['the classifier result: ', char_class])
    %disp (['the classifier + HMM result: ', char_yhmm])
    i = i+1; %new word start
    
    %error calcultation
    true = double(char(ytrue))-96;
    for k = 1:size(double(char(ytrue)), 1)
        if true(k) ~= ybar(k)
            err = err + 1;
        end
    end
end
disp (['error of classifier + HMM: ', num2str(err/i)])
