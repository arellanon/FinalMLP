clc, close all, clear all
fprintf('INICIANDO...\n');
%%% CARGA CONJUNTO DE ENTRENAMIENTO
%%% MATRIZ P[D1xD2]= P[5376x268]
%%% D1 = 5376: 256Hz * 3.5 seg * 6CH
%%% D2 =  268: Muestras
load('input/TRAIN.mat');

%%% CARGA TARGET DEL CONJUNTO DE ENTRENAMIENTO
%%% MATRIZ T[D1xD2] = T[1x268]
%%%  D1 =   1: CLASE ( 1 o -1 ) 
%%%  D2 = 268: MUESTRAS
%%% - LOS PRIMEROS  135 REGISTROS SON DE LA CLASE 0:  1
%%% - LOS RESTANTES 133 REGISTROS SON DE LA CLASE 1: -1
T = [ones(1,135) -ones(1,133)];
Q = size(P,2); % Cantidad de muestras

% VALORES INICIALES
%{
n1 = 73;  % Cantidad de neuronas capa oculta.
%%% INICIALIZACIÓN RANDOM DE LOS PESOS SINAPTICOS.
ep = 1;   % factor de variacion ( -1 a 1 )
W1 = 2*ep*rand(n1,size(P,1)) - ep;
b1 = 2*ep*rand(n1,1) - ep;  
W2 = 2*ep*rand(1,n1) - ep;
b2 = 2*ep*rand - ep;
%}
%%% INICIALIZACIÓN DE PESOS SINAPTICOS ALMACENADOS.
load('var/VARIABLES.mat'); % Pesos almacenados con 73 neuronas
alfa = 0.01; %Valor de ajuste

fprintf('PROCESANDO...\n');
for Epocas = 1:1000
    if mod(Epocas, 10) == 0
        fprintf('EPOCA: %d\n', Epocas);
    end
    sum = 0;
    for i = 1:Q
        q = randi(Q);
        % Propagación de la entrada hacia la salida
        a1 = tansig(W1*P(:,q) + b1);
        a2 = tansig(W2*a1 + b2);
        % Retropropagación de la sensibilidades
        e = T(q)-a2;
        s2 = -2*diag((1-a2^2))*e;
        s1 = diag((1-a1.^2))*W2'*s2;
        % Actualización de pesos sinapticos y polarizaciones
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
% Grafico del error cuadratico medio
figure, plot(emedio)
 
% Verificación de la respuesta.
CantidadOk = 0;
CantidadError = 0;
for q = 1:Q
    a(q) = tansig(W2*tansig(W1*P(:,q) + b1) + b2);
    if( (1 - a(q)*T(q)) < 0.5 )
        CantidadOk = CantidadOk + 1;
    else
        CantidadError = CantidadError + 1;
    end
end

% Grafico de la respuesta.
figure, hold on
plot(a,'r*')
plot([135, 135], [-1, 1],'b')
hold off;

fprintf('---------- CONJUNTO ENTRENAMIENTO ------------\n');
fprintf('PORCENTAJE CLASIFICADOS CORRECTAMENTE %3.2f %%\n', (CantidadOk * 100 / Q));
fprintf('PORCENTAJE CLASIFICADOS ERRONEAMENTE  %3.2f %%\n', (CantidadError * 100 / Q));


%%% CARGA CONJUNTO DE TEST
%%% MATRIZ P[D1xD2]= P[5376x293]
%%% D1 = 5376: 256Hz * 3.5 seg * 6CH
%%% D2 =  293: Muestras
load('input/TEST.mat');

%%% SE CARGA LA ETIQUETAS CON EL VALOR CORRESPONDIENTE PARA EL CONJUNTO
%%% DE TEST
load('input/LABELS_TEST.mat');
Q2 = size(TEST,2);           % Cantidad de muestras

% Verificación de la respuesta.
CantidadOkTest = 0;
CantidadErrorTest = 0;
for q = 1:Q2
    %%% Calculo el resultado del conjunto TEST en mi RNA-MLP entrenada
    RESULT = tansig(W2*tansig(W1*TEST(:,q) + b1) + b2);
    if( RESULT > 0) RESULTADO = 0; else RESULTADO = 1; end;
    %%% Verifico el valor obtenido contra el real
    if( (LABELS(q) - RESULTADO) == 0)
         CantidadOkTest = CantidadOkTest + 1;
    else
         CantidadErrorTest = CantidadErrorTest + 1;
    end
end

fprintf('-------------- CONJUNTO TEST -----------------\n');
fprintf('PORCENTAJE CLASIFICADOS CORRECTAMENTE %3.2f %%\n', (CantidadOkTest * 100 / Q2));
fprintf('PORCENTAJE CLASIFICADOS ERRONEAMENTE  %3.2f %%\n', (CantidadErrorTest * 100 / Q2));
fprintf('FINALIZADO\n');