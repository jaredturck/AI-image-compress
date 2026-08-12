#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct FileEntry {
    char *path;
    size_t size;
    uint64_t physical;
    size_t memory_offset;
};

static struct FileEntry *files;
static size_t file_count;
static size_t file_capacity;
static size_t total_size;

static double elapsed_seconds(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

static uint64_t get_physical_offset(const char *path)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return UINT64_MAX;

    size_t map_size = sizeof(struct fiemap) + sizeof(struct fiemap_extent);
    struct fiemap *map = calloc(1, map_size);

    map->fm_start = 0;
    map->fm_length = ~0ULL;
    map->fm_extent_count = 1;

    uint64_t physical = UINT64_MAX;

    if (ioctl(fd, FS_IOC_FIEMAP, map) == 0 && map->fm_mapped_extents)
        physical = map->fm_extents[0].fe_physical;

    free(map);
    close(fd);

    return physical;
}

static int collect_file(const char *path, const struct stat *stat_info, int type, struct FTW *ftw_info)
{
    (void)ftw_info;

    if (type != FTW_F || !S_ISREG(stat_info->st_mode))
        return 0;

    if (file_count == file_capacity) {
        file_capacity = file_capacity ? file_capacity * 2 : 65536;
        files = realloc(files, file_capacity * sizeof(*files));
    }

    files[file_count].path = strdup(path);
    files[file_count].size = stat_info->st_size;
    files[file_count].physical = get_physical_offset(path);

    total_size += stat_info->st_size;
    file_count++;

    return 0;
}

static int compare_files(const void *left, const void *right)
{
    const struct FileEntry *file_left = left;
    const struct FileEntry *file_right = right;

    if (file_left->physical < file_right->physical)
        return -1;

    if (file_left->physical > file_right->physical)
        return 1;

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s DIRECTORY\n", argv[0]);
        return 1;
    }

    printf("Scanning files...\n");
    nftw(argv[1], collect_file, 64, FTW_PHYS);

    printf("Found %zu files, %.1f MiB\n", file_count, total_size / 1024.0 / 1024.0);
    printf("Sorting by physical disk position...\n");
    qsort(files, file_count, sizeof(*files), compare_files);

    char *memory = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    size_t offset = 0;

    struct timespec start;
    struct timespec now;
    struct timespec last_update;

    clock_gettime(CLOCK_MONOTONIC, &start);
    last_update = start;

    for (size_t i = 0; i < file_count; i++) {
        files[i].memory_offset = offset;

        int fd = open(files[i].path, O_RDONLY | O_CLOEXEC | O_NOATIME);

        if (fd < 0 && errno == EPERM)
            fd = open(files[i].path, O_RDONLY | O_CLOEXEC);

        if (fd < 0) {
            perror(files[i].path);
            return 1;
        }

        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE);

        size_t remaining = files[i].size;
        size_t file_offset = 0;

        while (remaining) {
            ssize_t bytes = read(fd, memory + offset + file_offset, remaining);

            if (bytes <= 0) {
                perror(files[i].path);
                return 1;
            }

            file_offset += bytes;
            remaining -= bytes;
        }

        offset += files[i].size;
        close(fd);

        clock_gettime(CLOCK_MONOTONIC, &now);

        if (elapsed_seconds(last_update, now) >= 1.0 || i + 1 == file_count) {
            double seconds = elapsed_seconds(start, now);
            double percentage = (i + 1) * 100.0 / file_count;
            double mib_read = offset / 1024.0 / 1024.0;
            double throughput = seconds > 0.0 ? mib_read / seconds : 0.0;

            printf("\r%6.2f%% | %zu/%zu files | %.1f MiB | %.1fs | %.1f MiB/s",
                percentage, i + 1, file_count, mib_read, seconds, throughput);
            fflush(stdout);

            last_update = now;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &now);

    double seconds = elapsed_seconds(start, now);
    double mib = total_size / 1024.0 / 1024.0;

    printf("\n\nComplete\n");
    printf("Files: %zu\n", file_count);
    printf("Read: %.1f MiB\n", mib);
    printf("Time: %.2f seconds\n", seconds);
    printf("Throughput: %.1f MiB/s\n", mib / seconds);

    getchar();

    munmap(memory, total_size);

    return 0;
}
