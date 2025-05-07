#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <climits>
#include <chrono>

using namespace std;
using namespace std::chrono;

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int partition(int A[], int q, int r) {
    int x = A[q];
    int s = q;
    for (int i = q + 1; i <= r; i++) {
        if (A[i] <= x) {
            s++;
            swap(A[s], A[i]);
        }
    }
    swap(A[q], A[s]);
    return s;
}

void quicksort(int A[], int q, int r) {
    if (q < r) {
        int s = partition(A, q, r);
        quicksort(A, q, s - 1);
        quicksort(A, s + 1, r);
    }
}

bool readFromFile(const char* filename, int*& data, int& size) {
    ifstream file(filename);
    if (!file.is_open()) return false;

    file >> size;
    if (size <= 0) return false;

    data = new int[size];
    for (int i = 0; i < size; ++i) {
        if (!(file >> data[i])) {
            delete[] data;
            return false;
        }
    }

    file.close();
    return true;
}

void writeToFile(const char* filename, int* data, int size) {
    ofstream file(filename);
    for (int i = 0; i < size; ++i)
        file << data[i] << " ";
    file << "\n";
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string selected_file = "numere6.txt";

    int* data = nullptr;
    int original_n = 0;
    int n = 0;

    high_resolution_clock::time_point start_time, end_time;

    if (rank == 0) {
        if (!readFromFile(selected_file.c_str(), data, original_n)) {
            cerr << "Eroare la citirea fisierului\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = original_n;
        int padding = (size - (n % size)) % size;
        if (padding > 0) {
            int* new_data = new int[n + padding];
            for (int i = 0; i < n; ++i)
                new_data[i] = data[i];
            for (int i = 0; i < padding; ++i)
                new_data[n + i] = INT_MAX;

            delete[] data;
            data = new_data;
            n += padding;
        }

        start_time = high_resolution_clock::now();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    int* local_data = new int[local_n];

    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    quicksort(local_data, 0, local_n - 1);
    MPI_Gather(local_data, local_n, MPI_INT, data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        quicksort(data, 0, n - 1); 

        end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();

        int* clean_data = new int[original_n];
        for (int i = 0; i < original_n; ++i)
            clean_data[i] = data[i];

        string output_file = selected_file + "_sorted_parallel.txt";
        writeToFile(output_file.c_str(), clean_data, original_n);

        printf("Timpul total : %lld milisecunde\n", duration);

        delete[] clean_data;
        delete[] data;
    }

    delete[] local_data;
    MPI_Finalize();
    return 0;
}
