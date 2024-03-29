function gauss_seidel = matriz_temp(input)
    [linhas, coluna] = size(input);
    gauss_seidel = zeros(linhas, coluna);
    for i = 1:linhas
        for j = 1:linhas
            gauss_seidel(i, j)= (input(i, j) + (4)*(input(i, j) + 340 ))/(10);
        end
    end
end

