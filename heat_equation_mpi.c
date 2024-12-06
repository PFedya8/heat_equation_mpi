#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define L 10.0  // Длина стержня
#define T 11.0  // Общее время моделирования
#define A 1.0 // Коэффициент теплопроводности




int main(int argc, char *argv[]) {
    int rank, size;
    int Nx_global = 4096;  // Общее число узлов по пространству (можно изменить)
    int Nt;        // Число временных шагов (можно изменить)
    double dx;
    double dt;
    double C;              // Число Куранта
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Проверка, что число процессов является делителем Nx_global
    if (Nx_global % size != 0) {
        if (rank == 0) {
            printf("Число процессов должно быть делителем числа узлов по пространству (%d).\n", Nx_global);
        }
        MPI_Finalize();
        return -1;
    }

    int Nx_local = Nx_global / size;  // Число узлов на процесс
    dx = L / (Nx_global - 1);
    // printf("dx = %f\n", dx);
    dt = 0.3 * dx * dx / fabs(A);
    Nt = T / dt;
    // dt = T / Nt;
    // printf("dt = %f\n", dt);
    C = fabs(A) * dt / (dx * dx);

    if (rank == 0) {
        printf("Число процессов: %d\n", size);
        printf("Число узлов по x (глобальное): %d\n", Nx_global);
        printf("Число узлов по x (локальное): %d\n", Nx_local);
        printf("Число временных шагов: %d\n", Nt);
        printf("Число Куранта: %f\n", C);
        printf("dx = %f\n", dx);
        if (C > 0.5) {
            printf("Предупреждение: схема может быть неустойчивой (C = %f > 0.5).\n", C);
        }
    }

    // Выделение памяти для локальных массивов
    double *u_old = (double *)malloc((Nx_local + 2) * sizeof(double));
    double *u_new = (double *)malloc((Nx_local + 2) * sizeof(double));

    // Инициализация массивов
    for (int i = 1; i <= Nx_local; i++) {
        double x = dx * (rank * Nx_local + i - 1);
        // u_old[i] = sin(M_PI * x / L);
        u_old[i] = sin(x);
    }

    // Инициализация призрачных ячеек
    u_old[0] = 0.0;               // Левая призрачная ячейка (будет обновлена)
    u_old[Nx_local + 1] = 0.0;    // Правая призрачная ячейка (будет обновлена)

    double start_time, end_time;
    double t = 0.0;  // Текущее время
    double save_interval = T / 10.0;  // Сохраняем данные каждые 10% времени
    double next_save_time = save_interval;  // Время для следующего сохранения

    start_time = MPI_Wtime();

    // Основной цикл по времени
    for (int n = 0; t <= T; n++) {
        // Обмен граничными значениями с соседними процессами
        MPI_Request reqs[4];
        int req_count = 0;

        // Отправка и приём с левым соседом
        if (rank > 0) {
            MPI_Isend(&u_old[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&u_old[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        } else {
            u_old[0] = 0.0;  // Граничное условие на левой границе
        }

        // Отправка и приём с правым соседом
        if (rank < size - 1) {
            MPI_Isend(&u_old[Nx_local], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&u_old[Nx_local + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        } else {
            u_old[Nx_local + 1] = 0.0;  // Граничное условие на правой границе
        }

        // Ожидание завершения обмена
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        if (rank == 0 && fabs(u_old[0]) > 1e-6) {
            printf("Предупреждение: ошибка в граничном условии на левой границе (u[0] = %f).\n", u_old[0]);
        }
        if (rank == size - 1 && fabs(u_old[Nx_local + 1]) > 1e-6) {
            printf("Предупреждение: ошибка в граничном условии на правой границе (u[Nx_local + 1] = %f).\n", u_old[Nx_local + 1]);
        }

        // Обновление значений температуры во внутренних узлах
        for (int i = 1; i <= Nx_local; i++) {
            double f = 0.0;  // Правая часть, можно задать функцию при необходимости
            // double f = sin(dx * (rank * Nx_local + i - 1)) * exp(-t);
            u_new[i] = u_old[i] + C * (u_old[i + 1] - 2 * u_old[i] + u_old[i - 1]) + dt * f;
        }

        // Обновление массивов
        double *temp = u_old;
        u_old = u_new;
        u_new = temp;

        // Обновление текущего времени
        t += dt;

        // Сохранение промежуточных данных
        if (t >= next_save_time) {
            double *u_global = NULL;
            if (rank == 0) {
                u_global = (double *)malloc(Nx_global * sizeof(double));
            }

            // Собираем данные
            MPI_Gather(&u_old[1], Nx_local, MPI_DOUBLE, u_global, Nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                char filename[256];
                snprintf(filename, sizeof(filename), "result_t%.2f.dat", t);
                FILE *f = fopen(filename, "w");
                for (int i = 0; i < Nx_global; i++) {
                    double x = dx * i;
                    fprintf(f, "%f %f\n", x, u_global[i]);
                }
                fclose(f);
                printf("Результаты на t = %.2f сохранены в файл '%s'.\n", t, filename);
                free(u_global);
            }

            next_save_time += save_interval;
        }
    }
    end_time = MPI_Wtime();


    // Сбор результатов на процессе 0
    double *u_global = NULL;
    if (rank == 0) {
        u_global = (double *)malloc(Nx_global * sizeof(double));
    }

    // Собираем данные, исключая призрачные ячейки
    MPI_Gather(&u_old[1], Nx_local, MPI_DOUBLE, u_global, Nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
    if (rank == 0) {
        // Запись результатов в файл
        FILE *f = fopen("result.dat", "w");
        // print L, T, A, Nx_global, Nt
        FILE *f_param = fopen("param.dat", "w");
        fprintf(f_param, "%f %f %f %d %d\n", L, T, A, Nx_global, Nt);
        fclose(f_param);
        for (int i = 0; i < Nx_global; i++) {
            double x = dx * i;
            fprintf(f, "%f %f\n", x, u_global[i]);
        }
        fclose(f);
        printf("Результаты сохранены в файл 'result.dat'.\n");
        printf("Время выполнения: %f секунд\n", end_time - start_time);
        free(u_global);
    }

    free(u_old);
    free(u_new);

    MPI_Finalize();
    return 0;
}
