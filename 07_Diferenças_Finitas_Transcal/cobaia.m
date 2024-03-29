% Número de nós no eixo x e y
Nx = 4;
Ny = 6;

% Tamanhos dos intervalos de discretização
delta_x = 9;
delta_y = 4.5;

% Número máximo de iterações e critério de parada
maxIter = 1000;
tolerance = 1e-6;

% Inicialize a matriz de temperaturas (matriz de 24 termos)
T = zeros(Nx, Ny);

% Condições de Contorno
T(:, 1) = 0;
T(:, Ny) = 0;
T(1, :) = 85;
T(Nx, :) = 25;

% Inicialize a soma do calor total transferido pelo elemento central
totalHeatTransfer = 0;

% Coordenadas do nó central
centralNodeX = ceil(Nx / 2); % Nó central ao longo do eixo x
centralNodeY = ceil(Ny / 2); % Nó central ao longo do eixo y

% Realize as iterações do método de diferenças finitas
for iter = 1:maxIter
    T_antigo = T;
    
    for i = 2:Nx-1
        for j = 2:Ny-1
            % Calcule a taxa de calor em cada interface
            Q_x = (T(i+1, j) - T(i, j)) / delta_x;
            Q_y = (T(i, j+1) - T(i, j)) / delta_y;
            
            % Atualize a temperatura no nó
            T(i, j) = T(i, j) + (Q_x + Q_y) / 2;
        end
    end
    
    % Calcule o calor transferido pelo elemento central
    centralNodeHeatTransfer = -kappa * (T(centralNodeX, centralNodeY) - T_antigo(centralNodeX, centralNodeY));
    totalHeatTransfer = totalHeatTransfer + centralNodeHeatTransfer;
    
    % Verifique o critério de parada
    if max(abs(T(:) - T_antigo(:))) < tolerance
        break;
    end
end

% Exiba a matriz de temperaturas
disp('Matriz de Temperaturas:');
disp(T);

% Exiba o calor total transferido pelo elemento central
fprintf('Calor Total Transferido pelo Elemento Central: %.6f W\n', totalHeatTransfer);
