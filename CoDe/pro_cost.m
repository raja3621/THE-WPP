function [J grad out] = pro_cost(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                  val, X, Y, lambda)



Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 val, (hidden_layer_size + 1));


				 
m= size(X, 1);

n =  size(Theta1,2);
p =  size(Theta2,2);  

X=[ones(m,1) X];

act=(X*(Theta1)');
act1=[ones(m,1) act];

for i=1:(size(act,2)),
act(:,i)=pro_sigmoid(act(:,i));
end

act=[ones(m,1) act];

out=(act*(Theta2)');


for i=1:(size(out,2)),

out(:,i)=pro_sigmoid(out(:,i));
 
end

out1=log(out);
out2=log(1-out);


J=(sum(( -Y.*(out1))-(( 1-Y ).*out2)))/(m) +(((sum(sum((Theta1(:,(2:n)).^2),2)))+sum(sum((Theta2(:,(2:p)).^2),2)))*(lambda/(2*m)));


Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));


  for i=1:m,
  	 
	 delta3=(out(i) - (Y(i)));    %(1*1)

	 delta2=(((Theta2)' * delta3)) .* (sigmoidGradient(act1(i,:)'));
	 
	 
	 
	 delta2=delta2(2:length(delta2));   %(7*1) 
	 

	 
	 Delta1=Delta1+((delta2)*(X(i,:)));
	 Delta2=Delta2+((delta3)*(act(i,:)));

	
 end; 

theta1=Theta1;
theta2=Theta2;
theta1(:,1)=zeros(size(Theta1,1),1);
theta2(:,1)=zeros(size(Theta2,1),1);

Theta1_grad =(Delta1)/(m)+((lambda/m)*theta1); 
Theta2_grad =(Delta2)/(m)+((lambda/m)*theta2);
 
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end