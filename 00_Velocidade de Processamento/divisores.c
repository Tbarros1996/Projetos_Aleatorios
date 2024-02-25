#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int x, i;
    clock_t inicio;
    double tempo;
    int *interacao;
    double *tempos;

    printf("Entre com o Valor: ");
    scanf("%d", &x);

    interacao = (int*)malloc(x * sizeof(int));
    tempos = (double*)malloc(x * sizeof(double));

    inicio = clock();

    for (i = 1; i <= x; i++) {
        interacao[i - 1] = i - 1;
        tempo = (double)(clock() - inicio) / CLOCKS_PER_SEC;
        tempos[i - 1] = tempo;

        if (x % i == 0) {
            printf("%d\n", i);
        }
    }

    FILE *fp;
    fp = fopen("processamento.csv", "w");
    fprintf(fp, "Tempo de Processamento,Interacoes\n");
    for (i = 0; i < x; i++) {
        fprintf(fp, "%.2f,%d\n", tempos[i], interacao[i]);
    }
    fclose(fp);

    free(interacao);
    free(tempos);

    return 0;
}
