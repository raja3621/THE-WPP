clear ; close all; clc


input_layer_size  = 12; 
hidden_layer_size = 6;  
val=1;


initial_Theta1 = pro_rand_init(input_layer_size, hidden_layer_size);
initial_Theta2 = pro_rand_init(hidden_layer_size, 1);




inn_params = [initial_Theta1(:) ; initial_Theta2(:)];

x1= load('feautures_data.txt');
y1 = load('y_data.txt');


X=[x1 y1];

X= X(randperm(size(X,1)),:);

XC=X(241:320,1:((size(X,2)-1)));
YC=X(241:320,size(X,2));

XT=X(321:400,1:((size(X,2)-1)));
YT=X(321:400,size(X,2));

Y=X(1:240,size(X,2));

X=X(1:240,1:((size(X,2)-1)));


lambda = 3; 

[initial_cost,grad]= pro_cost(inn_params, input_layer_size, hidden_layer_size, ...
                   val, X, Y, lambda);
					

					fprintf(['INITIAL_COST_BRO \n %f \n'], initial_cost);

	
options = optimset('MaxIter', 50);
lambda=3;
%  You should also try different values of lambda

% Create "short hand" for the cost function to be minimized
costFunction = @(p) pro_cost(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   val, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction,inn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 val, (hidden_layer_size + 1));
		
	 			 
[pred,pred2,pred3,JC,JT,J] = predict(Theta1, Theta2, X,Y,XC,YC,XT,YT,lambda);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
fprintf(['\nCROSS_VALIDATION_SET_ERROR is %f\n'],JC);
fprintf('VALIDATION_Set Accuracy: %f\n', mean(double(pred2== YC)) * 100);
fprintf(['\nTEST_SET_ERROR is %f\n'],JT);						
fprintf('TEST_Set Accuracy: %f\n', mean(double(pred3== YT)) * 100);

fprintf('\n  DO_U_WANT_TO_TEST_ME\n');
str=input('yes or no\n','s');

id=0;
id=strcmpi(str,"no");

while(id==0),

inp=input('\n  ENTER THE MATCH DETAILS(FEAUTURES IN THE ORDER) TO PREDICT THE WINNING PERCENTAGE\n');

op=pro_result(inp,Theta1,Theta2);
fprintf(['\n   WINNING CHANCE:-->%f %\n'],op);	
fprintf('\n  DO_U_WANT_TO_TEST_ME\n');
str=input('yes or no\n','s');

id=strcmpi(str,"no");

end
					

