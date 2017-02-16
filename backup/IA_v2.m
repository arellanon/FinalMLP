clc, close all, clear all

%load('Traindata_0.txt');
%load('Traindata_1.txt');
load('P_TODO.mat');

T = [ones(1,135) -ones(1,133)];
Q = size(P,2);
%disp(Q);

%pause
% Valores iniciales

n1 = 73;  % 3 - 20
ep = 1;   %1

%{
W1 = 2*ep*rand(n1,size(P,1)) - ep; % n1: salida_a_capa_2 <- 896: entrada_red
b1 = 2*ep*rand(n1,1) - ep;  
W2 = 2*ep*rand(1,n1) - ep;   %  1: salida_red <- n1: entrada_desde_capa_1 
b2 = 2*ep*rand - ep;
%}

load('VARIABLES.mat');

% No usar logsig es muy lento, mejor sigmoid
alfa = 0.01;
for Epocas = 1:500
    sum = 0;
    for i = 1:Q
        q = randi(Q);
        % Propagaci�n de la entrada hacia la salida
        a1 = tansig(W1*P(:,q) + b1);
        a2 = tansig(W2*a1 + b2);
        % Retropropagaci�n de la sensibilidades
        e = T(q)-a2;
        s2 = -2*diag((1-a2^2))*e;
        s1 = diag((1-a1.^2))*W2'*s2;
        % Actualizaci�n de pesos sinapticos y polarizaciones
        W2 = W2 - alfa*s2*a1';
        b2 = b2 - alfa*s2;
        W1 = W1 - alfa*s1*P(:,q)';
        b1 = b1 - alfa*s1;  
        % Sumando el error cuadratico 
        sum = e^2 + sum;
    end
    % Error cuadratico medio
    emedio(Epocas) = sum/Q;
end
figure, plot(emedio)
 
% Verificaci�n grafica de la respuesta de la multicapa

for q = 1:Q
    a(q) = tansig(W2*tansig(W1*P(:,q) + b1) + b2);
end
figure, hold on
plot(a,'r*') 
plot([135, 135], [-1, 1],'b')
hold off;

%{
% Frontera de decisi�n
u = linspace(-15, 15, 50);
v = linspace(-15, 15, 50);
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = tansig(W2*tansig(W1*[u(i); v(j)] + b1) + b2);   
    end
end
figure, plot(P(1,1:100),P(2,1:100),'bo',P(1,101:200),P(2,101:200),'r*')
hold on, contour(u, v, z',[-0.9, 0.0, 0.9],'LineWidth', 2)  %Por que debo transponer a z
%}
