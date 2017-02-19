clc, close all, clear all
fprintf('INICIANDO...\n');
%%% CARGA CONJUNTO DE TEST
%%% MATRIZ P[D1xD2]= P[5376x293]
%%% D1 = 5376: 256Hz * 3.5 seg * 6CH
%%% D2 =  293: Muestras
load('input/TEST.mat');

%%% SE CARGA LA ETIQUETAS CON EL VALOR CORRESPONDIENTE PARA EL CONJUNTO
%%% DE TEST
load('input/LABELS_TEST.mat');

%%% SE CARGAN LOS PESOS SINAPTICOS OBTENIDOS EN EL ENTRENAMIENTO
load('var/VARIABLES.mat');
Q2 = size(TEST,2);           % Cantidad de muestras

% Verificación de la respuesta.
CantidadOkTest = 0;
CantidadErrorTest = 0;
for q = 1:Q2
    %% Calculo el resultado del conjunto TEST en mi RNA-MLP entrenada
    RESULT = tansig(W2*tansig(W1*TEST(:,q) + b1) + b2);
    if( RESULT > 0) RESULTADO = 0; else RESULTADO = 1; end;
    %% Verifico el valor obtenido contra el real
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