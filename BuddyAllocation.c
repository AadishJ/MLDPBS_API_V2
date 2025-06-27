#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <bits/pthreadtypes.h>
#include <string.h>

#define TOTAL_RAM_SIZE (512 * 1024 * 1024) // 512 MB
#define MAX_BLOCK_TYPES 10                 // Maximum number of different block sizes to support
#define ITERATION_COUNT 100000             // Total number of iterations for testing
#define NUM_THREADS 8                      // Number of threads

typedef struct
{
    uint8_t *pool;
    int pool_size;
    bool *free_blocks;
    int *allocated_sizes; // Track actual requested sizes for each block
    int block_size;
    pthread_spinlock_t lock; // Spin-lock for synchronization
} Partition;

typedef struct
{
    Partition partition;
} BuddyAllocator;

typedef struct
{
    BuddyAllocator *allocator;
    double *total_times;
    int *counts;
    int *block_sizes;
    int block_size_count;
    double *fragmentation_percentages; // Track fragmentation for each block size
} ThreadData;

typedef struct
{
    double internal_fragmentation_percentage;
    int allocated_blocks;
    long total_allocated_space;
    long total_requested_space;
} FragmentationStats;

void init(BuddyAllocator *allocator, int default_block_size);
void *allocateBuddySystem(BuddyAllocator *allocator, int size);
void *allocateBlock(Partition *partition, int size);
void deallocate(BuddyAllocator *allocator, void *ptr);
void cleanup(BuddyAllocator *allocator);
void *thread_worker(void *args);
int read_percentages(const char *filepath, double *percentages, int max_count);
FragmentationStats calculate_internal_fragmentation(BuddyAllocator *allocator);

void init(BuddyAllocator *allocator, int default_block_size)
{
    allocator->partition.pool_size = TOTAL_RAM_SIZE;
    allocator->partition.block_size = default_block_size;
    allocator->partition.pool = (uint8_t *)malloc(TOTAL_RAM_SIZE);

    int num_blocks = TOTAL_RAM_SIZE / default_block_size;
    allocator->partition.free_blocks = (bool *)malloc(num_blocks * sizeof(bool));
    allocator->partition.allocated_sizes = (int *)malloc(num_blocks * sizeof(int));
    pthread_spin_init(&allocator->partition.lock, 0);

    for (int j = 0; j < num_blocks; ++j)
    {
        allocator->partition.free_blocks[j] = true;
        allocator->partition.allocated_sizes[j] = 0; // Initialize to 0
    }
}

void *allocateBuddySystem(BuddyAllocator *allocator, int size)
{
    pthread_spin_lock(&allocator->partition.lock);
    void *ptr = allocateBlock(&allocator->partition, size);
    pthread_spin_unlock(&allocator->partition.lock);
    return ptr;
}

void *allocateBlock(Partition *partition, int size)
{
    int block_count = partition->pool_size / partition->block_size;
    for (int j = 0; j < block_count; ++j)
    {
        if (partition->free_blocks[j] && partition->block_size >= size)
        {
            partition->free_blocks[j] = false;
            partition->allocated_sizes[j] = size; // Store actual requested size
            return partition->pool + j * partition->block_size;
        }
    }
    return NULL;
}

void deallocate(BuddyAllocator *allocator, void *ptr)
{
    if (ptr >= (void *)allocator->partition.pool && ptr < (void *)(allocator->partition.pool + allocator->partition.pool_size))
    {
        int block_index = ((uint8_t *)ptr - allocator->partition.pool) / allocator->partition.block_size;
        allocator->partition.free_blocks[block_index] = true;
        allocator->partition.allocated_sizes[block_index] = 0; // Clear requested size
    }
}

FragmentationStats calculate_internal_fragmentation(BuddyAllocator *allocator)
{
    FragmentationStats stats = {0};

    pthread_spin_lock(&allocator->partition.lock);

    int block_count = allocator->partition.pool_size / allocator->partition.block_size;
    long total_allocated_space = 0;
    long total_requested_space = 0;
    int allocated_blocks = 0;

    for (int i = 0; i < block_count; i++)
    {
        if (!allocator->partition.free_blocks[i]) // Block is allocated
        {
            allocated_blocks++;
            total_allocated_space += allocator->partition.block_size;
            total_requested_space += allocator->partition.allocated_sizes[i];
        }
    }

    pthread_spin_unlock(&allocator->partition.lock);

    stats.allocated_blocks = allocated_blocks;
    stats.total_allocated_space = total_allocated_space;
    stats.total_requested_space = total_requested_space;

    // Calculate internal fragmentation percentage
    if (total_allocated_space > 0)
    {
        long internal_waste = total_allocated_space - total_requested_space;
        stats.internal_fragmentation_percentage = ((double)internal_waste / total_allocated_space) * 100.0;
    }
    else
    {
        stats.internal_fragmentation_percentage = 0.0;
    }

    return stats;
}

void cleanup(BuddyAllocator *allocator)
{
    free(allocator->partition.pool);
    free(allocator->partition.free_blocks);
    free(allocator->partition.allocated_sizes);
    pthread_spin_destroy(&allocator->partition.lock);
}

void *thread_worker(void *args)
{
    ThreadData *thread_data = (ThreadData *)args;
    BuddyAllocator *allocator = thread_data->allocator;
    int block_size_count = thread_data->block_size_count;
    int *block_sizes = thread_data->block_sizes;
    int iterations_per_thread = ITERATION_COUNT / NUM_THREADS;

    for (int b = 0; b < block_size_count; ++b)
    {
        clock_t start, end;
        double total_time = 0.0;
        start = clock();

        // Keep some allocations active to measure fragmentation
        void *persistent_ptrs[50];
        int persistent_count = 0;

        for (int i = 0; i < iterations_per_thread; ++i)
        {
            void *ptr = allocateBuddySystem(allocator, block_sizes[b]);
            if (ptr != NULL)
            {
                // Keep every 25th allocation to create fragmentation scenario
                if (i % 25 == 0 && persistent_count < 50)
                {
                    persistent_ptrs[persistent_count++] = ptr;
                }
                else
                {
                    deallocate(allocator, ptr);
                }
            }
        }

        end = clock();
        total_time += ((double)(end - start) / CLOCKS_PER_SEC) * 1000;

        // Calculate fragmentation with some blocks still allocated
        FragmentationStats frag_stats = calculate_internal_fragmentation(allocator);
        thread_data->fragmentation_percentages[b] = frag_stats.internal_fragmentation_percentage;

        // Clean up persistent allocations
        for (int i = 0; i < persistent_count; i++)
        {
            deallocate(allocator, persistent_ptrs[i]);
        }

        thread_data->total_times[b] += total_time;
        thread_data->counts[b] += 1;
    }
    return NULL;
}
// Replace the read_percentages function and modify main():
// Replace the read_percentages function and modify main():

int read_data_from_file(const char *filepath, double *percentages, int *block_sizes, int max_count)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        printf("Error: Cannot open file %s\n", filepath);
        return 0;
    }

    char line[256];
    int percentage_count = 0;
    int block_size_count = 0;
    int total_lines = 0;

    // First pass: count how many lines we have
    while (fgets(line, sizeof(line), file))
    {
        line[strcspn(line, "\n")] = 0; // Remove newline
        if (strlen(line) > 0)          // Skip empty lines
        {
            total_lines++;
        }
    }

    // Reset file pointer
    rewind(file);

    // Assume first half are percentages, second half are block sizes
    int expected_count = total_lines / 2;
    int current_line = 0;

    while (fgets(line, sizeof(line), file))
    {
        // Remove trailing newline
        line[strcspn(line, "\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0)
        {
            continue;
        }

        if (current_line < expected_count)
        {
            // Reading percentages
            char *endptr;
            double percentage = strtod(line, &endptr);
            if (endptr != line && percentage_count < max_count)
            {
                percentages[percentage_count] = percentage;
                percentage_count++;
            }
        }
        else
        {
            // Reading block sizes
            char *endptr;
            int block_size = (int)strtol(line, &endptr, 10);
            if (endptr != line && block_size_count < max_count)
            {
                block_sizes[block_size_count] = block_size;
                block_size_count++;
            }
        }
        current_line++;
    }

    fclose(file);

    // Return the minimum count to ensure we have matching pairs
    int min_count = (percentage_count < block_size_count) ? percentage_count : block_size_count;
    return min_count;
}
// Add this function to output system info in a parseable format
void output_system_info()
{
    printf("SYSTEM_INFO_START\n");
    printf("TOTAL_RAM_SIZE=%d\n", TOTAL_RAM_SIZE);
    printf("TOTAL_RAM_SIZE_MB=%d\n", TOTAL_RAM_SIZE / (1024 * 1024));
    printf("MAX_BLOCK_TYPES=%d\n", MAX_BLOCK_TYPES);
    printf("ITERATION_COUNT=%d\n", ITERATION_COUNT);
    printf("NUM_THREADS=%d\n", NUM_THREADS);
    printf("SYSTEM_INFO_END\n");
}

// Modify the main function to output system info at the beginning:
int main()
{
    // Output system information first for API parsing
    output_system_info();
    char file_path[256];

    // Try to find the file in different possible locations
    const char *possible_paths[] = {
        "predictions/percentages_batch_1.txt",
        "predictions/percentages_batch_2.txt",
        "predictions/percentages_batch_3.txt",
        "predictions/percentages.txt",
    };

    int path_found = 0;
    for (int i = 0; i < sizeof(possible_paths) / sizeof(possible_paths[0]); i++)
    {
        FILE *test = fopen(possible_paths[i], "r");
        if (test != NULL)
        {
            fclose(test);
            strcpy(file_path, possible_paths[i]);
            path_found = 1;
            break;
        }
    }
    if (!path_found)
    {
        printf("Error: percentages.txt not found in any expected location\n");
        return 1;
    }
    // Read data from file
    double percentages[MAX_BLOCK_TYPES];
    int block_sizes[MAX_BLOCK_TYPES];
    int count = read_data_from_file(file_path, percentages, block_sizes, MAX_BLOCK_TYPES);

    if (count == 0)
    {
        printf("No valid data found in file. Exiting.\n");
        return 1;
    }

    printf("Found %d entries in file\n", count);
    for (int i = 0; i < count; i++)
    {
        printf("Entry %d: %.2f%% - Block size: %d bytes\n", i + 1, percentages[i], block_sizes[i]);
    }

    // Initialize the allocator with the largest block size for simplicity
    int largest_block = block_sizes[0];
    for (int i = 1; i < count; i++)
    {
        if (block_sizes[i] > largest_block)
        {
            largest_block = block_sizes[i];
        }
    }

    BuddyAllocator allocator;
    init(&allocator, largest_block);

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    double total_times[MAX_BLOCK_TYPES] = {0};
    int counts[MAX_BLOCK_TYPES] = {0};
    double fragmentation_percentages[MAX_BLOCK_TYPES] = {0};

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        thread_data[i].allocator = &allocator;
        thread_data[i].total_times = total_times;
        thread_data[i].counts = counts;
        thread_data[i].block_sizes = block_sizes;
        thread_data[i].block_size_count = count;
        thread_data[i].fragmentation_percentages = fragmentation_percentages;
        pthread_create(&threads[i], NULL, thread_worker, &thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    printf("\nResults:\n");
    printf("Block Size\tAvg Time (ms)\tInternal Fragmentation (%%)\n");
    for (int b = 0; b < count; ++b)
    {
        double avg_time = total_times[b] / counts[b];
        double avg_fragmentation = fragmentation_percentages[b] / counts[b];
        printf("%d\t\t%.2f\t\t%.2f\n", block_sizes[b], avg_time, avg_fragmentation);
    }

    cleanup(&allocator);
    return 0;
}