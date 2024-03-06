/* 必要なヘッダファイルをインクルード */
#include <stdio.h>
#include <stdlib.h>  // Include the header file for malloc and free
#include <mpi.h>

#define LX 1.0        // 領域（X）のサイズ
#define LY 1.0        // 領域（Y）のサイズ
#define T 1.0         // シミュレーションを行う時間の上限
#define alpha 1.0     // 温度拡散率
#define T_SPAN 20     // 結果出力の頻度（適当に調整）

void print_data(int l_istart, int l_iend, int nprocs, int MX, int MY, int n, int my_rank, double dx, double dy, double dt, double *u)
{
    FILE *fp;
    char sfile[256];
    int i, j;

    sprintf(sfile, "1_17_%ddata_%06d.dat", my_rank, n);
    fp = fopen(sfile, "w");
    fprintf(fp, "#time = %lf\n", (double)n * dt);
    fprintf(fp, "#x y u\n");
    for (i = l_istart; i <= l_iend; i++)
    {
        for (j = 0; j <= MY; j++)
        {
            fprintf(fp, "%lf %lf %12lf\n", (double)i * dx, (double)j * dy, u[i * (MY + 1) + j]);
        }
        fprintf(fp, "\n");
    }
    if (my_rank == 0)
    {
        i = 0;
        for (j = 0; j <= MY; j++)
        {
            fprintf(fp, "%lf %lf %12lf\n", (double)i * dx, (double)j * dy, u[i * (MY + 1) + j]);
        }
        fprintf(fp, "\n");
    }
    if (my_rank == nprocs - 1)
    {
        i = MX;
        for (j = 0; j <= MY; j++)
        {
            fprintf(fp, "%lf %lf %12lf\n", (double)i * dx, (double)j * dy, u[i * (MY + 1) + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return;
}

int main(int argc, char *argv[])
{
    /* 変数の宣言（必要な変数は各自追加）*/
    double *u, *uu, dx, dy, dt;
    int i, j;
    int MX, MY, N;
    int n = 0;
    int nprocs, l_i, l_istart, l_iend, my_rank, prev_rank, next_rank;
    MPI_Status stat;

    /* 引数からMX と N の値を取得 */
    MX = 200;
    MY = 200;
    N = 160000;
    dx = LX / MX;
    dy = LY / MY;
    dt = T / N;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (nprocs == 1)
    {
        prev_rank = MPI_PROC_NULL;
        next_rank = MPI_PROC_NULL;
    }
    else if (my_rank == 0)
    {
        prev_rank = MPI_PROC_NULL;
        next_rank = (my_rank + 1) % nprocs;
    }
    else if (my_rank == nprocs - 1)
    {
        next_rank = MPI_PROC_NULL;
        prev_rank = (my_rank - 1 + nprocs) % nprocs;
    }
    else
    {
        prev_rank = (my_rank - 1 + nprocs) % nprocs;
        next_rank = (my_rank + 1) % nprocs;
    }

    l_i = (MX - 1) / nprocs;
    l_istart = (MX - 1) / nprocs * my_rank + 1;
    l_iend = (MX - 1) / nprocs * (my_rank + 1);
    if (my_rank == nprocs - 1)
    {
        l_i += (MX - 1) % nprocs;
        l_iend = l_istart + l_i - 1;
    }

    u = (double *)malloc(sizeof(double) * (MX + 1) * (MY + 1));
    uu = (double *)malloc(sizeof(double) * (MX + 1) * (MY + 1));

    for (i = 0; i <= MX; i++)
    {
        for (j = 0; j <= MY; j++)
        {
            if (j == 0)
            {
                u[i * (MY + 1) + j] = 20 * i * dx + 10;
            }
            else if (j == MY)
            {
                u[i * (MY + 1) + j] = 40 * i * dx + 40;
            }
            else if (i == 0)
            {
                u[i * (MY + 1) + j] = 30 * j * dy * j * dy + 10;
            }
            else if (i == MX)
            {
                u[i * (MY + 1) + j] = 50 * j * dy * j * dy + 30;
            }
            else
            {
                u[i * (MY + 1) + j] = 0.0;
            }
        }
    }

    double start_time = MPI_Wtime(); // Start time measurement

    for (n = 1; n <= N; n++)
    {

        MPI_Sendrecv(&u[l_istart * (MY + 1) + 1], MY - 1, MPI_DOUBLE, prev_rank, 0, &u[(l_iend + 1) * (MY + 1) + 1], MY - 1, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &stat);
        MPI_Sendrecv(&u[l_iend * (MY + 1) + 1], MY - 1, MPI_DOUBLE, next_rank, 1, &u[(l_istart - 1) * (MY + 1) + 1], MY - 1, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &stat);

        for (i = l_istart; i <= l_iend; i++)
        {
            for (j = 1; j < MY; j++)
            {
                uu[i * (MY + 1) + j] = u[i * (MY + 1) + j] + ((u[(i + 1) * (MY + 1) + j] - 2 * u[i * (MY + 1) + j] + u[(i - 1) * (MY + 1) + j]) / dx / dx + (u[i * (MY + 1) + j + 1] - 2 * u[i * (MY + 1) + j] + u[i * (MY + 1) + j - 1]) / dy / dy) * dt * alpha;
            }
        }
        for (i = l_istart; i <= l_iend; i++)
        {
            for (j = 1; j < MY; j++)
            {
                u[i * (MY + 1) + j] = uu[i * (MY + 1) + j];
            }
        }
    }

    double end_time = MPI_Wtime(); // End time measurement

    print_data(l_istart, l_iend, nprocs, MX, MY, N, my_rank, dx, dy, dt, u);

    free(uu);
    free(u);

    MPI_Finalize();

    if (my_rank == 0)
    {
        printf("実行時間: %lf 秒\n", end_time - start_time);
    }

    return 0;
}
